# src/vector_store.py
import hashlib
import json
import logging
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

import chromadb
import numpy as np
from chromadb import Settings

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get('database', {})
        self.client = None
        self.collection = None
        self._cache = {}

        # CORRECTION: Initialiser l'index avec des dicts vides pour chaque catégorie
        self._index = {
            'extension': defaultdict(list),
            'chunk_type': defaultdict(list),
            'file_path': defaultdict(list),
            'language': defaultdict(list)
        }

        self._embedding_function = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialise la base de données ChromaDB avec la nouvelle API"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            collection_name = self.db_config.get('collection_name', 'code_chunks')

            # NOUVELLE API ChromaDB
            settings = Settings()
            settings.anonymized_telemetry = False
            self.client = chromadb.PersistentClient(path=persist_path,settings=settings)

            # Configuration de la fonction d'embedding
            self._setup_embedding_function()

            # Créer ou récupérer la collection
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self._embedding_function
                )
                logger.info(f"✅ Collection '{collection_name}' chargée ({self.collection.count()} chunks)")
            except Exception as e:
                # Collection n'existe pas, la créer
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self._embedding_function,
                    metadata={
                        "description": "Code chunks embeddings for retro-engineering",
                        "created_at": datetime.now().isoformat(),
                        "version": "2.0",
                        "chunk_size": self.db_config.get('chunk_size', 512),
                        "overlap": self.db_config.get('overlap', 50)
                    }
                )
                logger.info(f"✅ Nouvelle collection '{collection_name}' créée")

            # Charger ou construire l'index
            self._load_index()

        except Exception as e:
            logger.error(f"❌ Erreur initialisation ChromaDB: {e}")
            logger.info("📋 Installation recommandée: pip install chromadb --upgrade")
            raise

    def _setup_embedding_function(self):
        """Configure la fonction d'embedding"""
        try:
            # Essayer d'utiliser SentenceTransformers si disponible
            from chromadb.utils import embedding_functions
            model_name = self.db_config.get('embedding_model', 'all-MiniLM-L6-v2')

            if model_name:
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
                logger.info(f"✅ Fonction d'embedding chargée: {model_name}")
            else:
                self._embedding_function = None
                logger.info("ℹ️  Utilisation de l'embedding par défaut de ChromaDB")

        except ImportError:
            logger.warning("⚠️  SentenceTransformer non disponible, utilisation embedding par défaut")
            self._embedding_function = None
        except Exception as e:
            logger.warning(f"⚠️  Erreur chargement embedding: {e}, utilisation par défaut")
            self._embedding_function = None

    def _load_index(self):
        """Charge ou construit un index rapide"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"  # CORRECTION: chemin complet

            if index_path.exists():
                with open(index_path, 'rb') as f:
                    loaded_index = pickle.load(f)

                # Restaurer la structure
                for key in ['extension', 'chunk_type', 'file_path', 'language']:
                    if key in loaded_index:
                        self._index[key] = defaultdict(list, loaded_index[key])
                    else:
                        self._index[key] = defaultdict(list)

                logger.info(f"✅ Index chargé ({len(self._index['extension'])} extensions)")
            else:
                logger.info("📊 Construction de l'index...")
                self._build_index()
        except Exception as e:
            logger.warning(f"⚠️  Impossible de charger l'index: {e}")
            self._init_index_structure()

    def _init_index_structure(self):
        """Initialise la structure de l'index"""
        self._index = {
            'extension': defaultdict(list),
            'chunk_type': defaultdict(list),
            'file_path': defaultdict(list),
            'language': defaultdict(list)
        }

    def _build_index(self, batch_size: int = 1000):
        """Construit l'index à partir de la collection"""
        try:
            total = self.collection.count()
            if total == 0:
                logger.info("ℹ️  Collection vide, pas d'index à construire")
                return

            # Réinitialiser l'index
            self._init_index_structure()

            for offset in range(0, total, batch_size):
                batch = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )

                if not batch.get('ids'):
                    continue

                for i, metadata in enumerate(batch['metadatas']):
                    if metadata and i < len(batch['ids']):
                        chunk_id = batch['ids'][i]

                        # Index par extension
                        ext = metadata.get('extension', 'unknown')
                        self._index['extension'][ext].append(chunk_id)

                        # Index par type de chunk
                        chunk_type = metadata.get('chunk_type', 'unknown')
                        self._index['chunk_type'][chunk_type].append(chunk_id)

                        # Index par chemin de fichier
                        file_path = metadata.get('file_path', '')
                        if file_path:
                            self._index['file_path'][file_path].append(chunk_id)

                        # Index par langage
                        language = metadata.get('language', 'unknown')
                        self._index['language'][language].append(chunk_id)

            self._save_index()
            logger.info(f"✅ Index construit: {len(self._index['extension'])} extensions, "
                        f"{len(self._index['chunk_type'])} types")

        except Exception as e:
            logger.error(f"❌ Erreur construction index: {e}")

    def _save_index(self):
        """Sauvegarde l'index sur disque"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"  # CORRECTION: chemin complet

            # S'assurer que le répertoire existe
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # Convertir defaultdict en dict normal pour la sérialisation
            serializable_index = {}
            for key, default_dict in self._index.items():
                serializable_index[key] = dict(default_dict)

            with open(index_path, 'wb') as f:
                pickle.dump(serializable_index, f)
            logger.debug("💾 Index sauvegardé")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde index: {e}")

    def add_chunks(self, file_info: Dict[str, Any], chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Ajoute des chunks avec la nouvelle API ChromaDB"""
        try:
            start_time = time.time()
            documents = []
            metadatas = []
            ids = []

            file_hash = hashlib.md5(file_info['path'].encode()).hexdigest()[:8]

            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_hash}_{i}_{int(time.time() * 1000)}"

                # Extraire le contenu du chunk
                content = self._extract_content_from_chunk(chunk)
                if isinstance(content, str):
                    content = content[:10000]
                else:
                    content = str(content)[:10000]

                documents.append(content)

                # Métadonnées enrichies
                enriched_metadata = {
                    'file_path': str(file_info.get('path', 'unknown')),
                    'relative_path': str(file_info.get('relative_path', 'unknown')),
                    'filename': str(file_info.get('filename', 'unknown')),
                    'extension': str(file_info.get('extension', 'unknown')),
                    'chunk_index': i,
                    'chunk_type': self._get_chunk_type(chunk),
                    'file_hash': file_hash,
                    'added_at': datetime.now().isoformat(),
                    'content_length': len(content),
                    'source': 'code_vectorizer'
                }

                # Ajouter les métadonnées du chunk si présentes
                if isinstance(chunk, dict) and 'metadata' in chunk:
                    chunk_metadata = chunk['metadata']
                    if isinstance(chunk_metadata, dict):
                        for key, value in chunk_metadata.items():
                            if key not in enriched_metadata and value is not None:
                                enriched_metadata[key] = value

                # Détection automatique
                auto_features = self._auto_detect_features(content, str(file_info.get('extension', '')))
                enriched_metadata.update(auto_features)

                metadatas.append(self._clean_metadata(enriched_metadata))
                ids.append(chunk_id)

                # CORRECTION: Mettre à jour l'index de manière sécurisée
                ext = str(file_info.get('extension', 'unknown'))
                self._index['extension'][ext].append(chunk_id)

                chunk_type = self._get_chunk_type(chunk)
                self._index['chunk_type'][chunk_type].append(chunk_id)

                file_path = str(file_info.get('path', ''))
                if file_path:
                    self._index['file_path'][file_path].append(chunk_id)

                # Indexer par langage si détecté
                language = auto_features.get('language', 'unknown')
                self._index['language'][language].append(chunk_id)

            # Vérifier que nous avons des documents à ajouter
            if not documents:
                logger.warning(f"⚠️ Aucun document à ajouter pour {file_info.get('filename', 'unknown')}")
                return []

            # Vérifier la cohérence des embeddings
            if isinstance(embeddings, np.ndarray) and len(embeddings) != len(documents):
                logger.warning(f"⚠️  Incohérence embeddings/documents: {len(embeddings)} vs {len(documents)}")
                # Ajuster si possible
                if len(embeddings) > len(documents):
                    embeddings = embeddings[:len(documents)]
                else:
                    # Dupliquer le dernier embedding ou créer des zéros
                    if len(embeddings) > 0:
                        last_embedding = embeddings[-1]
                        additional = np.array([last_embedding] * (len(documents) - len(embeddings)))
                        embeddings = np.vstack([embeddings, additional])
                    else:
                        # Créer des embeddings de zéros
                        embedding_dim = 384  # Dimension par défaut
                        embeddings = np.zeros((len(documents), embedding_dim))

            # Ajouter à ChromaDB
            batch_size = self.db_config.get('batch_size', 100)
            added_count = 0

            for batch_start in range(0, len(documents), batch_size):
                batch_end = min(batch_start + batch_size, len(documents))
                batch_ids = ids[batch_start:batch_end]
                batch_documents = documents[batch_start:batch_end]
                batch_metadatas = metadatas[batch_start:batch_end]

                try:
                    if self._embedding_function:
                        # Chroma calcule les embeddings
                        self.collection.add(
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )
                    else:
                        # Utiliser nos propres embeddings
                        batch_embeddings = embeddings[batch_start:batch_end]
                        self.collection.add(
                            embeddings=batch_embeddings.tolist(),
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            ids=batch_ids
                        )

                    added_count += len(batch_ids)

                except Exception as batch_error:
                    logger.error(f"❌ Erreur lors de l'ajout du lot {batch_start}-{batch_end}: {batch_error}")
                    # Essayer d'ajouter les documents un par un
                    for j in range(len(batch_ids)):
                        try:
                            if self._embedding_function:
                                self.collection.add(
                                    documents=[batch_documents[j]],
                                    metadatas=[batch_metadatas[j]],
                                    ids=[batch_ids[j]]
                                )
                            else:
                                single_embedding = [embeddings[batch_start + j].tolist()]
                                self.collection.add(
                                    embeddings=single_embedding,
                                    documents=[batch_documents[j]],
                                    metadatas=[batch_metadatas[j]],
                                    ids=[batch_ids[j]]
                                )
                            added_count += 1
                        except Exception as single_error:
                            logger.error(f"❌ Échec ajout chunk {batch_ids[j]}: {single_error}")
                            continue

            self._save_index()

            elapsed = time.time() - start_time
            logger.info(
                f"✅ {added_count}/{len(documents)} chunks ajoutés pour {file_info.get('filename', 'unknown')} en {elapsed:.2f}s")

            return ids[:added_count]

        except Exception as e:
            logger.error(f"❌ Erreur ajout chunks pour {file_info.get('path', 'unknown')}: {e}", exc_info=True)
            return []

    def _extract_content_from_chunk(self, chunk: Any) -> str:
        """Extrait le contenu d'un chunk de manière robuste"""
        if isinstance(chunk, dict):
            # Essayer différentes clés potentielles pour le contenu
            content_keys = ['content', 'text', 'code', 'body', 'data', 'value', 'document', 'source']

            for key in content_keys:
                if key in chunk and chunk[key] is not None:
                    content = chunk[key]
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, (dict, list)):
                        try:
                            return json.dumps(content, ensure_ascii=False)
                        except:
                            return str(content)
                    else:
                        return str(content)

            # Si aucune clé de contenu trouvée, convertir tout le dict
            try:
                return json.dumps(chunk, ensure_ascii=False)
            except:
                return str(chunk)

        elif isinstance(chunk, str):
            return chunk

        elif isinstance(chunk, (list, tuple)):
            try:
                return json.dumps(chunk, ensure_ascii=False)
            except:
                return str(chunk)

        else:
            return str(chunk)

    def _get_chunk_type(self, chunk: Any) -> str:
        """Détermine le type de chunk"""
        if isinstance(chunk, dict):
            return str(chunk.get('type', 'dict_chunk'))
        elif isinstance(chunk, str):
            return 'text_chunk'
        elif isinstance(chunk, list):
            return 'list_chunk'
        elif isinstance(chunk, tuple):
            return 'tuple_chunk'
        else:
            return str(type(chunk).__name__).lower()

    def _auto_detect_features(self, content: str, extension: str) -> Dict[str, Any]:
        """Détecte automatiquement des features dans le contenu"""
        features = {
            'language': self._detect_language(content, extension),
            'has_functions': bool(re.search(r'function\s+\w+|def\s+\w+|\w+\s*=\s*\([^)]*\)\s*=>', content)),
            'has_classes': bool(re.search(r'class\s+\w+', content)),
            'has_comments': bool(re.search(r'//|#|/\*|\*/|<!--|-->', content)),
            'line_count': len(content.split('\n')),
            'word_count': len(content.split()),
        }

        # Détection de patterns
        patterns = self._detect_patterns(content)
        if patterns:
            features['patterns'] = patterns

        # Détection de frameworks
        frameworks = self._detect_frameworks(content, extension)
        if frameworks:
            features['frameworks'] = frameworks

        return features

    def _detect_language(self, content: str, extension: str) -> str:
        """Détecte le langage de programmation"""
        language_map = {
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.md': 'markdown',
            '.sql': 'sql',
            '.xml': 'xml'
        }

        # D'abord par extension
        lang = language_map.get(extension.lower(), 'unknown')

        # Validation par contenu pour certains cas ambigus
        if lang == 'unknown' or lang == 'text':
            content_lower = content.lower()
            if 'def ' in content and ':' in content and not content.strip().startswith('//'):
                return 'python'
            elif 'function ' in content_lower or 'const ' in content_lower or 'let ' in content_lower:
                return 'javascript'
            elif 'public class' in content_lower or 'private ' in content_lower or 'system.out.println' in content_lower:
                return 'java'
            elif '#include <' in content_lower or 'std::' in content_lower:
                return 'cpp'
            elif '<?php' in content_lower or '$_' in content_lower:
                return 'php'
            elif '<!doctype html>' in content_lower or '<html>' in content_lower:
                return 'html'
            elif 'package ' in content_lower and 'import ' in content_lower:
                return 'java'

        return lang

    def _detect_patterns(self, content: str) -> List[str]:
        """Détecte les patterns de conception"""
        patterns = []
        content_lower = content.lower()

        pattern_checks = [
            ('singleton', [r'getinstance\(\)', r'instance\s*=\s*null', r'private\s+constructor']),
            ('factory', [r'factory\s+method', r'create[a-z][a-za-z]*\(', r'factory[a-z][a-za-z]*\s+class']),
            ('observer', [r'addobserver', r'notifyobservers', r'\.notify\(', r'implements.*observer']),
            ('decorator', [r'decorator', r'@[a-z][a-za-z]*\s*\(', r'extends.*decorator']),
            ('adapter', [r'adapter\s+class', r'implements.*adapter', r'adapts.*to']),
            ('strategy', [r'strategy\s+pattern', r'implements.*strategy', r'strategy[a-z][a-za-z]*\s+interface']),
            ('dependency_injection', [r'@inject', r'@autowired', r'injectable', r'dependencyinjection']),
            ('repository', [r'repository\s+pattern', r'extends.*repository', r'implements.*repository']),
            ('service', [r'service\s+layer', r'@service', r'service[a-z][a-za-z]*\s+class']),
            ('mvc', [r'controller', r'@controller', r'@restcontroller', r'model.*view.*controller'])
        ]

        for pattern_name, indicators in pattern_checks:
            for indicator in indicators:
                if re.search(indicator, content_lower):
                    patterns.append(pattern_name)
                    break

        return list(set(patterns))

    def _detect_frameworks(self, content: str, extension: str) -> List[str]:
        """Détecte les frameworks et bibliothèques"""
        frameworks = []
        content_lower = content.lower()

        framework_checks = [
            ('react', ['react', 'usestate', 'useeffect', 'react.createclass']),
            ('vue', ['vue', 'vue.component', 'v-model', 'v-for', 'v-if']),
            ('angular', ['angular', '@component', '@injectable', 'ngoninit']),
            ('express', ['express', 'app.get', 'app.post', 'express()']),
            ('django', ['django', 'from django', 'django.views']),
            ('flask', ['flask', '@app.route', 'flask(', 'from flask']),
            ('spring', ['@restcontroller', '@service', '@autowired', 'springbootapplication']),
            ('laravel', ['laravel', 'route::', 'eloquent', 'artisan']),
            ('nextjs', ['next/', 'next/router', 'getserversideprops']),
            ('nuxtjs', ['nuxt/', 'nuxt.config', 'asyncdata']),
            ('nestjs', ['@nestjs', 'nestjs', '@controller', '@module']),
            ('fastapi', ['fastapi', '@app.get', 'fastapi()']),
            ('jquery', ['jquery', '$(']),
            ('bootstrap', ['bootstrap', 'data-bs-', 'btn-primary']),
            ('tailwind', ['tailwind', 'class=".*-\\[.*\\]']),
        ]

        for framework_name, indicators in framework_checks:
            for indicator in indicators:
                if indicator in content_lower:
                    frameworks.append(framework_name)
                    break

        return list(set(frameworks))

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               where_filter: Optional[Dict] = None, include: List[str] = None) -> List[Dict[str, Any]]:
        """Recherche vectorielle avec la nouvelle API"""
        try:
            # Paramètres par défaut pour include
            if include is None:
                include = ["documents", "metadatas", "distances"]

            # Recherche dans ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=where_filter,
                include=include
            )

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"❌ Erreur recherche vectorielle: {e}")
            return []

    def search_by_text(self, query_text: str, top_k: int = 5,
                       where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Recherche par texte (Chroma calcule l'embedding automatiquement)"""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"❌ Erreur recherche textuelle: {e}")
            return []

    def _format_search_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Formate les résultats de recherche"""
        formatted_results = []

        if not results.get('documents'):
            return formatted_results

        for i in range(len(results['documents'][0])):
            try:
                result = {
                    'content': results['documents'][0][i],
                    'metadata': self._deserialize_metadata(results['metadatas'][0][i]),
                    'similarity': 1 - (results['distances'][0][i] if results['distances'] else 0),
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i]
                }

                # Ajouter les embeddings si disponibles
                if 'embeddings' in results and results['embeddings']:
                    result['embedding'] = results['embeddings'][0][i]

                formatted_results.append(result)

            except Exception as e:
                logger.warning(f"⚠️  Erreur formatage résultat {i}: {e}")
                continue

        # Trier par similarité (décroissant)
        formatted_results.sort(key=lambda x: x['similarity'], reverse=True)

        return formatted_results

    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 100,
                           include: List[str] = None) -> List[Dict[str, Any]]:
        """Recherche par métadonnées"""
        try:
            if include is None:
                include = ["documents", "metadatas"]

            # CORRECTION: Utiliser l'index de manière sécurisée
            use_index = False
            chunk_ids = []

            # Vérifier si le filtre est simple et indexé
            if len(filters) == 1:
                key = list(filters.keys())[0]
                value = filters[key]

                # Vérifier si la clé est dans notre index
                if key in self._index:
                    # CORRECTION: Accéder correctement à l'index
                    index_dict = self._index[key]
                    if value in index_dict:
                        chunk_ids = index_dict[value][:limit]
                        use_index = True

            if use_index and chunk_ids:
                # Utiliser l'index
                results = self.collection.get(
                    ids=chunk_ids,
                    include=include
                )
                return self._format_get_results(results)
            else:
                # Fallback: recherche directe
                results = self.collection.get(
                    where=filters,
                    limit=limit,
                    include=include
                )

                return self._format_get_results(results)

        except Exception as e:
            logger.error(f"❌ Erreur recherche par métadonnées: {e}")
            return []

    def _format_get_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Formate les résultats de get()"""
        formatted = []

        if not results.get('documents'):
            return formatted

        for i in range(len(results['documents'])):
            try:
                formatted.append({
                    'content': results['documents'][i],
                    'metadata': self._deserialize_metadata(results['metadatas'][i]),
                    'id': results['ids'][i]
                })
            except Exception as e:
                logger.warning(f"⚠️  Erreur formatage résultat get {i}: {e}")
                continue

        return formatted

    def get_chunks_by_file(self, file_path: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère tous les chunks d'un fichier spécifique"""
        chunks = self.search_by_metadata({'file_path': file_path}, limit)

        # Trier par chunk_index si disponible
        try:
            chunks.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
        except:
            pass

        return chunks

    def get_chunks_by_extension(self, extension: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les chunks par extension de fichier"""
        return self.search_by_metadata({'extension': extension}, limit)

    def get_chunks_by_language(self, language: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les chunks par langage"""
        return self.search_by_metadata({'language': language}, limit)

    def get_all_chunks(self, limit: int = 2000, offset: int = 0) -> List[Dict[str, Any]]:
        """Récupère tous les chunks avec pagination"""
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            return self._format_get_results(results)

        except Exception as e:
            logger.error(f"❌ Erreur récupération chunks: {e}")
            return []

    def get_chunks_generator(self, batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """Générateur pour parcourir les chunks par lots"""
        total = self.collection.count()

        for offset in range(0, total, batch_size):
            chunks = self.get_all_chunks(limit=batch_size, offset=offset)
            if chunks:
                yield chunks
            else:
                break

    def analyze_collection(self) -> Dict[str, Any]:
        """Analyse complète de la collection"""
        try:
            total = self.collection.count()

            if total == 0:
                return {
                    'total_chunks': 0,
                    'message': 'Collection vide'
                }

            extensions = defaultdict(int)
            chunk_types = defaultdict(int)
            languages = defaultdict(int)

            sample_size = min(1000, total)
            sample = self.collection.get(
                limit=sample_size,
                include=["metadatas"]
            )

            for metadata in sample['metadatas']:
                if metadata:
                    extensions[metadata.get('extension', 'unknown')] += 1
                    chunk_types[metadata.get('chunk_type', 'unknown')] += 1
                    languages[metadata.get('language', 'unknown')] += 1

            return {
                'total_chunks': total,
                'extensions': dict(extensions),
                'chunk_types': dict(chunk_types),
                'languages': dict(languages),
                'collection_name': self.collection.name,
                'collection_metadata': self.collection.metadata,
                'index_stats': self.get_index_stats()
            }

        except Exception as e:
            logger.error(f"❌ Erreur analyse collection: {e}")
            return {'error': str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'index"""
        try:
            stats = {}

            for key in self._index.keys():
                index_dict = self._index[key]
                if isinstance(index_dict, defaultdict):
                    index_dict = dict(index_dict)

                stats[f'{key}_count'] = len(index_dict)
                total_ids = sum(len(ids) for ids in index_dict.values())
                stats[f'{key}_total_ids'] = total_ids

                # Top 5 des valeurs les plus fréquentes
                if index_dict:
                    top_items = sorted(
                        index_dict.items(),
                        key=lambda x: len(x[1]),
                        reverse=True
                    )[:5]

                    stats[f'{key}_top_5'] = [
                        {'value': val, 'count': len(ids)}
                        for val, ids in top_items
                    ]

            return stats

        except Exception as e:
            logger.error(f"❌ Erreur statistiques index: {e}")
            return {'error': str(e)}

    def debug_index(self):
        """Affiche des informations de debug sur l'index"""
        print("\n🔍 DEBUG INDEX")
        print("=" * 50)

        for key in self._index.keys():
            print(f"\n📁 {key.upper()}:")
            index_dict = self._index[key]

            if isinstance(index_dict, defaultdict):
                index_dict = dict(index_dict)

            print(f"  Nombre d'entrées: {len(index_dict)}")

            # Afficher quelques exemples
            if index_dict:
                sample_items = list(index_dict.items())[:3]
                for subkey, ids in sample_items:
                    print(f"  - '{subkey}': {len(ids)} IDs")

                    # Afficher quelques IDs
                    if ids:
                        print(f"    Exemples d'IDs: {ids[:2]}")

        print("\n" + "=" * 50)

    def delete_chunks_by_file(self, file_path: str) -> int:
        """Supprime tous les chunks d'un fichier"""
        try:
            # Utiliser l'index pour trouver les IDs
            chunk_ids = []
            if 'file_path' in self._index and file_path in self._index['file_path']:
                chunk_ids = self._index['file_path'][file_path]

            if not chunk_ids:
                # Fallback: recherche dans la collection
                chunks = self.get_chunks_by_file(file_path, limit=10000)
                chunk_ids = [chunk['id'] for chunk in chunks]

            if chunk_ids:
                self.collection.delete(ids=chunk_ids)

                # Nettoyer l'index
                for key in self._index:
                    index_dict = self._index[key]
                    for subkey in list(index_dict.keys()):
                        # Filtrer les IDs supprimés
                        index_dict[subkey] = [id for id in index_dict[subkey] if id not in chunk_ids]
                        # Supprimer les clés vides
                        if not index_dict[subkey]:
                            del index_dict[subkey]

                self._save_index()
                logger.info(f"✅ Supprimé {len(chunk_ids)} chunks pour {file_path}")
                return len(chunk_ids)

            return 0

        except Exception as e:
            logger.error(f"❌ Erreur suppression chunks: {e}")
            return 0

    def update_chunk(self, chunk_id: str, content: str = None, metadata: Dict = None) -> bool:
        """Met à jour un chunk existant"""
        try:
            update_data = {'ids': [chunk_id]}

            if content is not None:
                update_data['documents'] = [content]

            if metadata is not None:
                update_data['metadatas'] = [self._clean_metadata(metadata)]

            self.collection.update(**update_data)
            logger.debug(f"✅ Chunk {chunk_id} mis à jour")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur mise à jour chunk: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la base"""
        try:
            return {
                'total_chunks': self.collection.count(),
                'collection_name': self.collection.name,
                'collection_metadata': self.collection.metadata,
                'embedding_function': 'custom' if self._embedding_function else 'default',
                'persist_path': self.db_config.get('path', './vector_db'),
                'index_stats': self.get_index_stats()
            }
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {'error': str(e)}

    def _clean_metadata(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie les métadonnées pour ChromaDB"""
        cleaned = {}

        for key, value in metadata_dict.items():
            if value is None:
                # Valeurs par défaut selon le type de champ
                if key in ['is_processed', 'is_valid']:
                    cleaned[key] = False
                elif key in ['chunk_index', 'line_count', 'content_length', 'word_count']:
                    cleaned[key] = 0
                elif key in ['similarity', 'complexity_score']:
                    cleaned[key] = 0.0
                elif key in ['patterns', 'frameworks']:
                    cleaned[key] = []
                else:
                    cleaned[key] = ""
            elif isinstance(value, (list, dict)):
                # Sérialiser les structures complexes
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except:
                    cleaned[key] = str(value)[:500]
            elif isinstance(value, (int, float, bool, str)):
                cleaned[key] = value
            else:
                # Conversion en string pour les autres types
                cleaned[key] = str(value)[:500]

        return cleaned

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Désérialise les métadonnées"""
        deserialized = {}

        if not metadata:
            return deserialized

        for key, value in metadata.items():
            if value is None:
                deserialized[key] = None
            elif isinstance(value, str):
                # Essayer de désérialiser le JSON
                try:
                    stripped = value.strip()
                    if (stripped.startswith('[') and stripped.endswith(']')) or \
                            (stripped.startswith('{') and stripped.endswith('}')):
                        deserialized[key] = json.loads(stripped)
                    else:
                        deserialized[key] = value
                except:
                    deserialized[key] = value
            else:
                deserialized[key] = value

        return deserialized

    def save(self):
        """Sauvegarde la base et l'index"""
        try:
            # CORRECTION: ChromaDB PersistentClient gère automatiquement la persistance
            # On sauvegarde seulement notre index personnalisé
            self._save_index()
            logger.debug("💾 Index sauvegardé")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")

    def reset(self, confirm: bool = False) -> bool:
        """Réinitialise la collection"""
        if not confirm:
            logger.warning("⚠️  Réinitialisation annulée: confirmation requise")
            return False

        try:
            collection_name = self.db_config.get('collection_name', 'code_chunks')
            self.client.delete_collection(collection_name)

            # Recréer la collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self._embedding_function,
                metadata={
                    "description": "Reset collection",
                    "reset_at": datetime.now().isoformat(),
                    "version": "2.0"
                }
            )

            # Réinitialiser l'index
            self._init_index_structure()
            self._save_index()

            logger.warning(f"♻️  Collection '{collection_name}' réinitialisée")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur réinitialisation: {e}")
            return False

    def export_collection(self, output_path: str = None) -> str:
        """Exporte la collection au format JSON"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"./exports/collection_export_{timestamp}.json"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            chunks = self.get_all_chunks(limit=10000)

            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'total_chunks': len(chunks),
                    'collection_name': self.collection.name,
                    'collection_metadata': self.collection.metadata
                },
                'chunks': chunks
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"📤 Collection exportée vers {output_path} ({len(chunks)} chunks)")
            return output_path

        except Exception as e:
            logger.error(f"❌ Erreur export: {e}")
            return ""

    def close(self):
        """Ferme proprement la connexion"""
        try:
            self.save()  # Sauvegarde juste l'index
            logger.info("👋 VectorStore fermé")
        except Exception as e:
            logger.error(f"❌ Erreur fermeture: {e}")


def get_existing_chunks(vector_store: VectorStore = None, config: Dict[str, Any] = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Récupère tous les chunks existants de la base vectorielle
    """
    try:
        # Si vector_store n'est pas fourni, en créer un
        if vector_store is None:
            if config is None:
                config_loader = ConfigLoader("config/vectorization_config.yaml")
                config = config_loader.config
            vector_store = VectorStore(config)

        # Méthode 1: Utiliser une recherche avec un embedding neutre pour récupérer tous les chunks
        # Créer un embedding de recherche "neutre" (vecteur de zéros)
        dummy_embedding = np.zeros(384)  # Taille par défaut de all-MiniLM-L6-v2

        # Récupérer un grand nombre de chunks
        results = vector_store.search(
            query_embedding=dummy_embedding,
            top_k=limit
        )

        logger.info(f"✅ {len(results)} chunks récupérés depuis la base vectorielle")
        return results

    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des chunks: {e}")
        return []


def get_chunks_by_file_pattern(vector_store: VectorStore, pattern: str,
                               limit: int = 200) -> List[Dict[str, Any]]:
    """
    Récupère les chunks correspondant à un pattern de fichier
    """
    try:
        # Pour ChromaDB, nous devons utiliser where pour filtrer par métadonnées
        # Cette méthode nécessite d'accéder directement à l'API ChromaDB
        collection = vector_store.collection

        # Récupérer tous les éléments avec filtre
        results = collection.get(
            limit=limit,
            where={"file_path": {"$contains": pattern}}
        )

        chunks = []
        for i in range(len(results['documents'])):
            chunks.append({
                'content': results['documents'][i],
                'metadata': vector_store._deserialize_metadata(results['metadatas'][i]),
                'id': results['ids'][i]
            })

        logger.info(f"✅ {len(chunks)} chunks trouvés pour le pattern '{pattern}'")
        return chunks

    except Exception as e:
        logger.error(f"❌ Erreur recherche par pattern '{pattern}': {e}")
        return []


def get_chunks_by_type(vector_store: VectorStore, chunk_type: str,
                       limit: int = 200) -> List[Dict[str, Any]]:
    """
    Récupère les chunks par type (js_function, js_class, java_class, etc.)
    """
    try:
        collection = vector_store.collection

        results = collection.get(
            limit=limit,
            where={"chunk_type": {"$eq": chunk_type}}
        )

        chunks = []
        for i in range(len(results['documents'])):
            chunks.append({
                'content': results['documents'][i],
                'metadata': vector_store._deserialize_metadata(results['metadatas'][i]),
                'id': results['ids'][i]
            })

        logger.info(f"✅ {len(chunks)} chunks de type '{chunk_type}' trouvés")
        return chunks

    except Exception as e:
        logger.error(f"❌ Erreur recherche par type '{chunk_type}': {e}")
        return []


def get_all_chunks_direct(vector_store: VectorStore, limit: int = 2000) -> List[Dict[str, Any]]:
    """
    Récupère directement tous les chunks via l'API ChromaDB (méthode alternative)
    """
    try:
        collection = vector_store.collection

        # Récupérer tous les documents sans filtre
        results = collection.get(limit=limit)

        chunks = []
        for i in range(len(results['documents'])):
            chunks.append({
                'content': results['documents'][i],
                'metadata': vector_store._deserialize_metadata(results['metadatas'][i]),
                'id': results['ids'][i]
            })

        logger.info(f"✅ {len(chunks)} chunks récupérés directement depuis ChromaDB")
        return chunks

    except Exception as e:
        logger.error(f"❌ Erreur récupération directe des chunks: {e}")
        return []
