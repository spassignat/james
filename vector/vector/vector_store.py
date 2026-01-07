import hashlib
import json
import logging
import pickle
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
import numpy as np
import time
from chromadb import Settings, EmbeddingFunction
from chromadb.utils import embedding_functions

from models.search_intent import SearchIntent

logger = logging.getLogger(__name__)


class VectorStore:
    """Gestionnaire de base de données vectorielle pour les chunks de code"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get("database", {})
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self._index = defaultdict(dict)
        self._embedding_function: Optional[EmbeddingFunction] = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialise la base de données ChromaDB"""
        try:
            persist_path = self.db_config.get("path", "./vector_db")
            collection_name = self.db_config.get("collection_name", "code_chunks")

            # Configuration ChromaDB
            settings = Settings(
                anonymized_telemetry=False,
                persist_directory=persist_path,
                chroma_server_host='localhost',
                chroma_server_http_port=8000,
                chroma_server_grpc_port=50051
            )

            # Créer le répertoire de persistance
            Path(persist_path).mkdir(parents=True, exist_ok=True)

            # Initialiser le client
            self.client = chromadb.PersistentClient(
                path=persist_path,
                settings=settings
            )

            # Configurer la fonction d'embedding
            self._setup_embedding_function()

            # Créer ou charger la collection
            self._setup_collection(collection_name)

            # Charger l'index
            self._load_index()

            logger.info(f"✅ VectorStore initialisé (collection: {collection_name})")

        except Exception as e:
            logger.error(f"❌ Erreur initialisation VectorStore: {e}")
            raise

    def _setup_collection(self, collection_name: str):
        """Configure la collection ChromaDB"""
        try:
            # Essayer de charger une collection existante
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            )
            logger.info(f"✅ Collection '{collection_name}' chargée ({self.collection.count()} chunks)")

        except ValueError:
            # Créer une nouvelle collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self._embedding_function,
                metadata={
                    "description": "Code chunks embeddings",
                    "created_at": datetime.now().isoformat(),
                    "version": "2.0",
                    "chunk_size_limit": 5000,
                    "embedding_model": self.db_config.get("embedding_model", "all-MiniLM-L6-v2")
                }
            )
            logger.info(f"✅ Nouvelle collection '{collection_name}' créée")

        except Exception as e:
            logger.error(f"❌ Erreur configuration collection: {e}")
            raise

    def _setup_embedding_function(self):
        """Configure la fonction d'embedding"""
        try:
            model_name = self.db_config.get("embedding_model", "all-MiniLM-L6-v2")
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device=self.db_config.get("embedding_device", "cpu")
            )
            logger.info(f"✅ Fonction d'embedding configurée: {model_name}")

        except Exception as e:
            logger.warning(f"⚠️ Erreur configuration embedding, utilisation par défaut: {e}")
            # Fallback vers une fonction d'embedding simple
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def _load_index(self):
        """Charge l'index depuis le disque"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"

            if index_path.exists():
                with open(index_path, 'rb') as f:
                    loaded_index = pickle.load(f)

                # Restaurer la structure avec defaults
                index_keys = ['extension', 'chunk_type', 'file_path', 'language','processed_date',
                              'chunk_size', 'category', 'function_name', 'class_name']

                for key in index_keys:
                    if key in loaded_index:
                        self._index[key] = defaultdict(list, loaded_index[key])
                    else:
                        self._index[key] = defaultdict(list)

                logger.info(f"✅ Index chargé: {len(self._index['extension'])} extensions, "
                            f"{len(self._index['chunk_type'])} types, "
                            f"{sum(len(v) for v in self._index['file_path'].values())} fichiers")

            else:
                logger.info(f"⚠️ Index non trouvé, création d'un index vide")
                self._init_index_structure()

        except Exception as e:
            logger.error(f"❌ Erreur chargement index: {e}")
            self._init_index_structure()

    def _init_index_structure(self):
        """Initialise la structure vide de l'index"""
        self._index = {
            'extension': defaultdict(list),
            'chunk_type': defaultdict(list),
            'file_path': defaultdict(list),
            'language': defaultdict(list),
            'chunk_size': defaultdict(list),
            'category': defaultdict(list),
            'function_name': defaultdict(list),
            'class_name': defaultdict(list),
            'processed_date': defaultdict(list),
            'chunk_id_map': {}  # Map chunk_id -> metadata summary
        }

    def persist_index(self):
        """Sauvegarde l'index sur disque"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # Convertir les defaultdict en dict pour la sérialisation
            serializable_index = {}
            for key, value in self._index.items():
                if isinstance(value, defaultdict):
                    serializable_index[key] = dict(value)
                else:
                    serializable_index[key] = value

            with open(index_path, 'wb') as f:
                pickle.dump(serializable_index, f)

            logger.debug("💾 Index sauvegardé")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde index: {e}")

    def add_chunks(self, file_path: str, chunks: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None):
        """Ajoute des chunks au vector store avec embeddings optionnels (interface compatible)"""
        try:
            logger.info(f"📥 Ajout de {len(chunks)} chunks pour {file_path}")

            # Préparer les données pour ChromaDB
            ids = []
            documents = []
            metadatas = []
            chunk_embeddings = []

            for i, chunk in enumerate(chunks):
                # Générer l'ID du chunk
                chunk_id = self._generate_chunk_id(chunk, file_path, i)
                ids.append(chunk_id)

                # Extraire le contenu
                content = self._extract_chunk_content(chunk)
                documents.append(content)

                # Préparer les métadonnées
                metadata = self._prepare_chunk_metadata(chunk, file_path, i, content)
                metadatas.append(metadata)

                # Gérer les embeddings
                if embeddings is not None and i < len(embeddings):
                    # Utiliser les embeddings fournis
                    if hasattr(embeddings[i], 'tolist'):
                        chunk_embeddings.append(embeddings[i].tolist())
                    else:
                        chunk_embeddings.append(embeddings[i])
                elif 'embedding' in chunk and chunk['embedding'] is not None:
                    # Utiliser l'embedding du chunk
                    chunk_embeddings.append(chunk['embedding'])

                # Mettre à jour l'index local
                self._update_index(chunk_id, metadata, content)

            # Ajouter à ChromaDB
            if chunk_embeddings and len(chunk_embeddings) == len(ids):
                # Avec embeddings
                self._add_batch_to_chromadb(ids, documents, metadatas, chunk_embeddings)
            else:
                # Sans embeddings (ChromaDB générera)
                self._add_batch_to_chromadb(ids, documents, metadatas)

            logger.info(f"✅ {len(chunks)} chunks ajoutés pour {file_path}")

        except Exception as e:
            logger.error(f"❌ Erreur ajout chunks pour {file_path}: {e}")
            raise

    def _generate_chunk_id(self, chunk: Dict[str, Any], file_path: str, index: int) -> str:
        """Génère un ID unique pour un chunk"""
        # Utiliser l'ID du chunk s'il existe
        if 'id' in chunk and chunk['id']:
            return chunk['id']

        # Générer un ID basé sur le contenu et le fichier
        content = chunk.get('content', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        timestamp = int(time.time() * 1000) % 1000000

        return f"{Path(file_path).stem}_{index}_{content_hash}_{file_hash}_{timestamp:06d}"

    def _extract_chunk_content(self, chunk: Dict[str, Any]) -> str:
        """Extrait et valide le contenu d'un chunk"""
        content = chunk.get('content', '')

        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                content = ''

        # Limiter la taille pour ChromaDB
        max_length = self.db_config.get('max_chunk_size', 5000)
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        return content

    def _prepare_chunk_metadata(self, chunk: Dict[str, Any], file_path: str,
                                index: int, content: str) -> Dict[str, Any]:
        """Prépare les métadonnées pour ChromaDB"""
        # Métadonnées de base
        metadata = {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'chunk_index': index,
            'chunk_id': chunk.get('id', ''),
            'chunk_type': chunk.get('type', 'unknown'),
            'chunk_size': len(content),
            'processed_at': datetime.now().isoformat(),
            'content_hash': hash(content) % 1000000
        }

        # Ajouter les métadonnées du chunk
        chunk_metadata = chunk.get('metadata', {})
        if isinstance(chunk_metadata, dict):
            for key, value in chunk_metadata.items():
                if value is not None and key not in metadata:
                    metadata[key] = self._clean_metadata_value(value)

        # Nettoyer pour ChromaDB
        return self._clean_metadata_dict(metadata)

    def _clean_metadata_value(self, value: Any) -> Any:
        """Nettoie une valeur de métadonnée pour ChromaDB"""
        if value is None:
            return ""
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, dict)):
            try:
                # Sérialiser en JSON avec limite de taille
                json_str = json.dumps(value, ensure_ascii=False)
                if len(json_str) > 1000:
                    return json_str[:1000] + "..."
                return json_str
            except:
                return str(value)[:1000]
        else:
            return str(value)[:500]

    def _clean_metadata_dict(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie un dictionnaire de métadonnées pour ChromaDB"""
        cleaned = {}
        for key, value in metadata_dict.items():
            cleaned[key] = self._clean_metadata_value(value)
        return cleaned

    def _update_index(self, chunk_id: str, metadata: Dict[str, Any], content: str):
        """Met à jour l'index interne avec le nouveau chunk"""
        try:
            # Indexer par extension
            extension = metadata.get('extension', 'unknown')
            self._index['extension'][extension].append(chunk_id)

            # Indexer par type de chunk
            chunk_type = metadata.get('chunk_type', 'unknown')
            self._index['chunk_type'][chunk_type].append(chunk_id)

            # Indexer par fichier
            file_path = metadata.get('file_path', 'unknown')
            self._index['file_path'][file_path].append(chunk_id)

            # Indexer par langue si présente
            language = metadata.get('language', 'unknown')
            if language != 'unknown':
                self._index['language'][language].append(chunk_id)

            # Indexer par taille de chunk
            chunk_size = metadata.get('chunk_size', 0)
            size_category = self._get_size_category(chunk_size)
            self._index['chunk_size'][size_category].append(chunk_id)

            # Indexer par catégorie si présente
            category = metadata.get('category', 'unknown')
            if category != 'unknown':
                self._index['category'][category].append(chunk_id)

            # Indexer par nom de fonction si présent
            function_name = metadata.get('function_name', 'unknown')
            if function_name != 'unknown':
                self._index['function_name'][function_name].append(chunk_id)

            # Indexer par nom de classe si présent
            class_name = metadata.get('class_name', 'unknown')
            if class_name != 'unknown':
                self._index['class_name'][class_name].append(chunk_id)

            # Indexer par date de traitement
            processed_date = metadata.get('processed_at', '').split('T')[0]
            if processed_date:
                self._index['processed_date'][processed_date].append(chunk_id)

            # Stocker un résumé dans la map
            self._index['chunk_id_map'][chunk_id] = {
                'file_path': file_path,
                'chunk_type': chunk_type,
                'extension': extension,
                'size': chunk_size,
                'language': language
            }

            logger.debug(f"📝 Chunk {chunk_id} indexé")

        except Exception as e:
            print(traceback.format_exc())
            logger.warning(f"⚠️ Erreur indexation chunk {chunk_id}: {e}")

    def _get_size_category(self, size: int) -> str:
        """Catégorise la taille d'un chunk"""
        if size < 100:
            return 'tiny'
        elif size < 500:
            return 'small'
        elif size < 2000:
            return 'medium'
        elif size < 5000:
            return 'large'
        else:
            return 'huge'

    def _add_batch_to_chromadb(self, ids: List[str], documents: List[str],
                               metadatas: List[Dict[str, Any]],
                               embeddings: Optional[List[List[float]]] = None):
        """Ajoute un batch de chunks à ChromaDB"""
        try:
            if embeddings and len(embeddings) == len(ids):
                # Avec embeddings pré-calculés
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Sans embeddings (ChromaDB les calcule)
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

            logger.debug(f"✅ Batch de {len(ids)} chunks ajouté à ChromaDB")

        except Exception as e:
            logger.error(f"❌ Erreur ajout batch à ChromaDB: {e}")
            raise

    def search(self, query_embedding: str, top_k: int = 5,
               threshold: float = 0.5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Recherche des chunks similaires à un embedding donné.
        """
        try:
            # Convertir l'embedding en liste
            if hasattr(query_embedding, 'tolist'):
                embedding_list = query_embedding.tolist()
            else:
                embedding_list = query_embedding

            # Préparer les filtres
            where_filters = self._build_chroma_filters(filters) if filters else None

            # Effectuer la recherche
            results = self.collection.query(
                query_embeddings=[embedding_list],
                n_results=top_k,
                where=where_filters,
                include=["documents", "metadatas", "distances"]
            )

            # Formater les résultats
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1.0 - distance

                    if similarity >= threshold:
                        result = {
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'id': results['ids'][0][i],
                            'similarity': float(similarity),
                            'distance': float(distance)
                        }
                        formatted_results.append(result)

            logger.debug(f"🔍 Recherche: {len(formatted_results)} résultats (seuil: {threshold})")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []

    def search_by_text(self, query: str, top_k: int = 5,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Recherche par texte.
        """
        try:
            where_filters = self._build_chroma_filters(filters) if filters else None

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filters,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i],
                        'distance': float(results['distances'][0][i]) if results['distances'] else 0,
                        'similarity': 1.0 - (float(results['distances'][0][i]) if results['distances'] else 0)
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur recherche par texte: {e}")
            return []

    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Recherche par métadonnées.
        """
        try:
            where_filters = self._build_chroma_filters(filters)

            results = self.collection.get(
                where=where_filters,
                limit=limit,
                include=["documents", "metadatas"]
            )

            formatted_results = []
            for i in range(len(results['documents'])):
                result = {
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i]
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur recherche par métadonnées: {e}")
            return []

    def _build_chroma_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Construit des filtres compatibles avec ChromaDB"""
        chroma_filters = {}

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                chroma_filters[key] = value
            elif isinstance(value, list):
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    chroma_filters[key] = {"$in": value}
            elif isinstance(value, dict):
                # Support des opérateurs ChromaDB
                chroma_filters[key] = value

        return chroma_filters

    def get_all_chunks(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Retourne tous les chunks sous forme de dictionnaire"""
        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )

            chunks = []
            for i in range(len(results["documents"])):
                chunk = {
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "id": results["ids"][i],
                }
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"❌ Erreur récupération chunks: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk spécifique par son ID"""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )

            if results['documents']:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'id': results['ids'][0]
                }
            return None

        except Exception as e:
            logger.error(f"❌ Erreur récupération chunk {chunk_id}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales"""
        try:
            total_chunks = self.collection.count()

            # Récupérer un échantillon pour analyser les métadonnées
            sample_size = min(1000, total_chunks)
            sample = self.collection.get(limit=sample_size, include=["metadatas"])

            # Analyser les extensions
            extensions = defaultdict(int)
            chunk_types = defaultdict(int)
            languages = defaultdict(int)

            for metadata in sample['metadatas']:
                ext = metadata.get('extension', 'unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                lang = metadata.get('language', 'unknown')

                extensions[ext] += 1
                chunk_types[chunk_type] += 1
                languages[lang] += 1

            return {
                'total_chunks': total_chunks,
                'collection_name': self.collection.name,
                'collection_metadata': self.collection.metadata,
                'embedding_model': self.db_config.get('embedding_model', 'unknown'),
                'persist_path': self.db_config.get('path', './vector_db'),
                'sample_analysis': {
                    'extensions': dict(sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]),
                    'chunk_types': dict(sorted(chunk_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                    'languages': dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10])
                },
                'index_stats': self.get_index_stats()
            }

        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {'error': str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'index interne"""
        try:
            stats = {
                'total_chunks_indexed': len(self._index['chunk_id_map']),
                'files_indexed': len(self._index['file_path']),
                'extensions_indexed': len(self._index['extension']),
                'chunk_types_indexed': len(self._index['chunk_type']),
                'languages_indexed': len(self._index['language']),
                'top_extensions': [],
                'top_chunk_types': [],
                'top_languages': []
            }

            # Top extensions
            if self._index['extension']:
                top_ext = sorted(self._index['extension'].items(),
                                 key=lambda x: len(x[1]), reverse=True)[:5]
                stats['top_extensions'] = [{'extension': ext, 'count': len(ids)} for ext, ids in top_ext]

            # Top chunk types
            if self._index['chunk_type']:
                top_types = sorted(self._index['chunk_type'].items(),
                                   key=lambda x: len(x[1]), reverse=True)[:5]
                stats['top_chunk_types'] = [{'type': typ, 'count': len(ids)} for typ, ids in top_types]

            # Top languages
            if self._index['language']:
                top_langs = sorted(self._index['language'].items(),
                                   key=lambda x: len(x[1]), reverse=True)[:5]
                stats['top_languages'] = [{'language': lang, 'count': len(ids)} for lang, ids in top_langs]

            return stats

        except Exception as e:
            logger.error(f"❌ Erreur statistiques index: {e}")
            return {'error': str(e)}

    @staticmethod
    def build_query(intent: SearchIntent) -> str:
        return f"""
        Goal: {intent.goal}
        Domains: {intent.domain}
        Focus: {', '.join(intent.focus)}
        Depth: {intent.depth}
        """

    def search_intent(self, intent: SearchIntent, top_k: int = None) -> Dict[str, List]:
        """
        Recherche selon une intention de recherche.
        """
        try:
            # Déterminer top_k basé sur la profondeur
            if top_k is None:
                if intent.depth == "high":
                    top_k = 8
                elif intent.depth == "medium":
                    top_k = 4
                else:
                    top_k = 2

            # Construire la requête
            query_text = intent.get_search_query() if hasattr(intent, 'get_search_query') else intent.goal

            # Construire les filtres
            where_filters = {}
            if hasattr(intent, 'file_type') and intent.file_type:
                where_filters['extension'] = intent.file_type
            if hasattr(intent, 'language') and intent.language:
                where_filters['language'] = intent.language

            # Effectuer la recherche

            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filters if where_filters else None,
                include=["documents", "metadatas", "distances"]
            )

            # Formater les résultats
            formatted_results = {
                'query': query_text,
                'intent': intent.to_dict() if hasattr(intent, 'to_dict') else intent.__dict__,
                'results': []
            }

            if results['documents']:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i],
                        'distance': float(distance),
                        'similarity': 1.0 - float(distance)
                    }
                    formatted_results['results'].append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erreur recherche intent: {e}")
            return {'error': str(e), 'results': []}

    def delete_file_chunks(self, file_path: str) -> int:
        """
        Supprime tous les chunks d'un fichier.
        """
        try:
            # Rechercher les IDs des chunks du fichier
            results = self.collection.get(
                where={'file_path': file_path},
                include=[]
            )

            if results['ids']:
                # Supprimer les chunks
                self.collection.delete(ids=results['ids'])

                # Mettre à jour l'index
                self._remove_from_index(file_path, results['ids'])

                logger.info(f"🗑️  {len(results['ids'])} chunks supprimés pour {file_path}")
                return len(results['ids'])

            return 0

        except Exception as e:
            logger.error(f"❌ Erreur suppression chunks: {e}")
            return 0

    def _remove_from_index(self, file_path: str, chunk_ids: List[str]):
        """Supprime des chunks de l'index interne"""
        # Supprimer de tous les index
        for key in self._index:
            if isinstance(self._index[key], defaultdict):
                for index_key, id_list in list(self._index[key].items()):
                    # Filtrer les IDs à supprimer
                    self._index[key][index_key] = [id for id in id_list if id not in chunk_ids]

                    # Supprimer la clé si la liste est vide
                    if not self._index[key][index_key]:
                        del self._index[key][index_key]

        # Supprimer de la map
        for chunk_id in chunk_ids:
            if chunk_id in self._index['chunk_id_map']:
                del self._index['chunk_id_map'][chunk_id]

    def cleanup(self):
        """Nettoie les ressources"""
        try:
            self.persist_index()
            logger.info("🧹 VectorStore nettoyé")
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage VectorStore: {e}")

    def export_chunks(self, output_path: str, limit: int = None) -> bool:
        """Exporte les chunks vers un fichier JSON"""
        try:
            chunks = self.get_all_chunks(limit or 10000)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"✅ {len(chunks)} chunks exportés vers {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur export chunks: {e}")
            return False
