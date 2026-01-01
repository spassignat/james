# src/code_vectorizer.py
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config.file_scanner import FileScanner
from vector.chunk.registry_chunk import ChunkStrategyRegistry
from vector.vector_store import VectorStore
import numpy as np
logger = logging.getLogger(__name__)


class CodeVectorizer:
    def __init__(self, config: Dict[str, Any], analyzer):
        self.config = config
        self.analyzer = analyzer
        self.vectorization_config = config.get('vectorization', {})

        # Initialisation des composants
        self.file_scanner = FileScanner(config.get('project', {}))
        self.chunk_strategy_registry = ChunkStrategyRegistry(self.vectorization_config)
        self.vector_store = VectorStore(self.vectorization_config)

        # Modèle d'embedding
        self.model = self._initialize_model()

        logger.info("CodeVectorizer initialisé")

    def _initialize_model(self) -> SentenceTransformer:
        """Initialise le modèle d'embedding"""
        model_name = self.vectorization_config.get('model_name', 'all-MiniLM-L6-v2')
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Modèle d'embedding chargé: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    def vectorize_project(self) -> Dict[str, Any]:
        """Vectorise l'ensemble du projet avec suivi de progression"""
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'chunks_by_type': {},
            'files_by_extension': {}
        }

        logger.info("🚀 Début de la vectorisation du projet...")

        for file_info in self.file_scanner.scan_project():
            stats['total_files'] += 1

            try:
                file_chunks = self._process_file(file_info)
                if file_chunks:
                    stats['processed_files'] += 1
                    stats['total_chunks'] += len(file_chunks)

                    # Statistiques par type de chunk et extension
                    extension = file_info['extension']
                    stats['files_by_extension'][extension] = stats['files_by_extension'].get(extension, 0) + 1

                    for chunk in file_chunks:
                        chunk_type = chunk.get('type', 'unknown')
                        stats['chunks_by_type'][chunk_type] = stats['chunks_by_type'].get(chunk_type, 0) + 1

                    if stats['processed_files'] % 20 == 0:
                        logger.debug(f"Fichier vectorisé: {file_info['path']} ({len(file_chunks)} chunks)")
                else:
                    stats['failed_files'] += 1
                    logger.debug(f"Aucun chunk créé pour: {file_info['path']}")

            except Exception as e:
                print(traceback.format_exc())
                stats['failed_files'] += 1
                logger.error(f"Erreur traitement fichier {file_info['path']}: {e}")

            # Log de progression
            if stats['total_files'] % 50 == 0:
                self._log_progress(stats)

        # Log final et sauvegarde
        self._log_final_stats(stats)
        self.vector_store.save()

        return stats


    # src/code_vectorizer.py (méthodes corrigées)

    def _process_file(self, file_info: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Traite un fichier individuel - CORRIGÉ"""
        try:
            # 1. Analyse du fichier
            analysis = self.analyzer.analyze(file_info['path'])
            if not analysis or analysis.get('error'):
                logger.warning(f"Analyse échouée pour: {file_info['path']}")
                return None

            # 2. Création des chunks
            chunks = self.chunk_strategy_registry.create_chunks(
                file_info['extension'],
                analysis,
                file_info
            )

            if not chunks:
                logger.warning(f"Aucun chunk créé pour: {file_info['path']}")
                return None

            # 3. Validation et normalisation des chunks - CORRECTION: S'assurer que chunks est une liste
            if not isinstance(chunks, list):
                logger.warning(f"Les chunks ne sont pas une liste pour {file_info['path']}, type: {type(chunks)}")
                # Essayer de convertir en liste
                if isinstance(chunks, dict):
                    chunks = [chunks]
                else:
                    chunks = list(chunks) if hasattr(chunks, '__iter__') else [chunks]

            normalized_chunks = self._normalize_chunks(chunks, file_info)
            if not normalized_chunks:
                return None

            # 4. Vectorisation et stockage
            return self._vectorize_and_store_chunks(file_info, normalized_chunks)

        except Exception as e:
            logger.error(f"Erreur traitement fichier {file_info['path']}: {e}", exc_info=True)
            # Fallback : traitement basique du fichier
            return self._fallback_file_processing(file_info)

    def _normalize_chunks(self, chunks: List, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise la structure des chunks - CORRIGÉ"""
        normalized_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                normalized_chunk = self._normalize_single_chunk(chunk, i, file_info)
                if normalized_chunk and isinstance(normalized_chunk, dict):
                    normalized_chunks.append(normalized_chunk)
                else:
                    logger.warning(f"Chunk {i} normalisé invalide pour {file_info['path']}")
            except Exception as e:
                logger.warning(f"Erreur normalisation chunk {i} pour {file_info['path']}: {e}")
                continue

        return normalized_chunks

    def _normalize_single_chunk(self, chunk: Any, index: int, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalise un chunk individuel - CORRIGÉ"""
        try:
            # CORRECTION: Gérer différents types de chunks
            if isinstance(chunk, dict):
                # Vérifier si c'est déjà un chunk normalisé
                if 'content' in chunk and isinstance(chunk['content'], str):
                    # C'est déjà un chunk normalisé, ajouter des métadonnées
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}

                    # Ajouter les métadonnées de base
                    chunk['metadata'].update({
                        'file_path': file_info.get('path', ''),
                        'relative_path': file_info.get('relative_path', ''),
                        'filename': file_info.get('filename', ''),
                        'extension': file_info.get('extension', ''),
                        'chunk_index': index,
                        'chunk_id': f"{Path(file_info.get('path', 'unknown')).stem}_{index}"
                    })

                    # S'assurer que 'type' existe
                    if 'type' not in chunk:
                        chunk['type'] = chunk.get('metadata', {}).get('chunk_type', 'normalized_dict')

                    return chunk
                else:
                    # C'est un dict mais pas un chunk normalisé, essayer d'extraire le contenu
                    content = self._extract_content_from_dict(chunk)
                    if content:
                        return self._create_standard_chunk(content, 'dict_chunk', file_info, index, chunk)
                    else:
                        # Fallback: convertir tout le dict
                        return self._create_standard_chunk(
                            json.dumps(chunk, ensure_ascii=False),
                            'json_chunk',
                            file_info,
                            index
                        )

            elif isinstance(chunk, str):
                return self._create_standard_chunk(chunk, 'text', file_info, index)

            elif isinstance(chunk, (list, tuple)):
                content = json.dumps(chunk, ensure_ascii=False)
                return self._create_standard_chunk(content, 'list_chunk', file_info, index)

            else:
                # Autres types (int, float, etc.)
                content = str(chunk)
                return self._create_standard_chunk(content, 'fallback', file_info, index)

        except Exception as e:
            logger.error(f"Erreur normalisation chunk: {e}")
            # Fallback ultime
            return self._create_standard_chunk(
                f"Chunk error: {str(e)[:200]}",
                'error',
                file_info,
                index
            )

    def _create_standard_chunk(self, content: Any, chunk_type: str,
                               file_info: Dict[str, Any], index: int,
                               original_chunk: Optional[Dict] = None) -> Dict[str, Any]:
        """Crée un chunk standardisé - CORRIGÉ"""
        try:
            # S'assurer que le contenu est une string
            if not isinstance(content, str):
                content = self._convert_to_string(content)

            # Limiter la taille
            content = content[:5000] if len(content) > 5000 else content

            # Métadonnées de base
            metadata = {
                'file_path': file_info.get('path', ''),
                'relative_path': file_info.get('relative_path', ''),
                'filename': file_info.get('filename', ''),
                'extension': file_info.get('extension', ''),
                'chunk_index': index,
                'chunk_id': f"{Path(file_info.get('path', 'unknown')).stem}_{index}",
                'chunk_type': chunk_type
            }

            # Ajouter des métadonnées de l'original si disponibles
            if original_chunk and isinstance(original_chunk, dict):
                # Copier les métadonnées simples
                simple_keys = ['line_start', 'line_end', 'language', 'category',
                               'function_name', 'class_name', 'module']

                for key in simple_keys:
                    if key in original_chunk and original_chunk[key] is not None:
                        value = original_chunk[key]
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)[:200]

            chunk = {
                'content': content,
                'type': chunk_type,
                'metadata': metadata
            }

            return chunk

        except Exception as e:
            logger.error(f"Erreur création chunk standard: {e}")
            # Fallback minimal
            return {
                'content': f"Error creating chunk: {str(e)[:200]}",
                'type': 'error',
                'metadata': {
                    'file_path': file_info.get('path', 'error'),
                    'chunk_index': index,
                    'error': str(e)[:100]
                }
            }

    def _vectorize_and_store_chunks(self, file_info: Dict[str, Any],
                                    chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vectorise et stocke les chunks - CORRIGÉ"""
        try:
            # CORRECTION: Vérifier que tous les chunks ont du contenu
            valid_chunks = []
            chunk_contents = []

            for chunk in chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                    if content and isinstance(content, str) and content.strip():
                        valid_chunks.append(chunk)
                        chunk_contents.append(content)
                    else:
                        logger.warning(f"Chunk sans contenu valide pour {file_info['path']}")
                else:
                    logger.warning(f"Chunk invalide pour {file_info['path']}: {type(chunk)}")

            if not valid_chunks:
                logger.warning(f"Aucun chunk valide pour {file_info['path']}")
                return None

            # Vectorisation par lots
            batch_size = self.vectorization_config.get('batch_size', 50)
            all_embeddings = []

            for i in range(0, len(chunk_contents), batch_size):
                batch = chunk_contents[i:i + batch_size]
                try:
                    batch_embeddings = self.model.encode(batch)
                    all_embeddings.extend(batch_embeddings)
                except Exception as batch_error:
                    logger.error(f"Erreur vectorisation lot {i}: {batch_error}")
                    # Créer des embeddings de repli
                    if all_embeddings:
                        # Dupliquer le dernier embedding réussi
                        last_embedding = all_embeddings[-1]
                        all_embeddings.extend([last_embedding] * len(batch))
                    else:
                        # Créer des embeddings de zéros
                        embedding_dim = self.model.get_sentence_embedding_dimension()
                        zero_embedding = np.zeros(embedding_dim)
                        all_embeddings.extend([zero_embedding] * len(batch))

            # Convertir en numpy array
            try:
                embeddings = np.array(all_embeddings)
            except Exception as e:
                logger.error(f"Erreur création array embeddings: {e}")
                # Fallback: créer un array de zéros
                if all_embeddings and hasattr(all_embeddings[0], '__len__'):
                    embedding_dim = len(all_embeddings[0])
                else:
                    embedding_dim = 384  # Dimension par défaut
                embeddings = np.zeros((len(all_embeddings), embedding_dim))

            # Stockage dans la base vectorielle
            self.vector_store.add_chunks(file_info, valid_chunks, embeddings)

            logger.info(f"✅ {len(valid_chunks)} chunks vectorisés pour {file_info['filename']}")
            return valid_chunks

        except Exception as e:
            logger.error(f"Erreur vectorisation pour {file_info['path']}: {e}", exc_info=True)
            return None

    def _normalize_chunks(self, chunks: List, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalise la structure des chunks pour assurer la cohérence"""
        normalized_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                normalized_chunk = self._normalize_single_chunk(chunk, i, file_info)
                if normalized_chunk:
                    normalized_chunks.append(normalized_chunk)
            except Exception as e:
                logger.warning(f"Erreur normalisation chunk {i} pour {file_info['path']}: {e}")
                continue

        return normalized_chunks

    def _normalize_single_chunk(self, chunk: Any, index: int, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalise un chunk individuel"""
        # Si le chunk est déjà un dict avec la structure attendue
        if isinstance(chunk, dict) and 'content' in chunk:
            return self._enrich_chunk_metadata(chunk, file_info, index)

        # Si le chunk est un dict mais sans structure standard
        elif isinstance(chunk, dict):
            content = self._extract_content_from_dict(chunk)
            if content:
                return self._create_standard_chunk(content, 'normalized_dict', file_info, index, chunk)

        # Si le chunk est une string
        elif isinstance(chunk, str):
            return self._create_standard_chunk(chunk, 'text', file_info, index)

        # Autres types
        else:
            content = str(chunk)
            return self._create_standard_chunk(content, 'fallback', file_info, index)

        return None
    def _convert_to_string(self, content: Any) -> str:
        """Convertit n'importe quel contenu en string de manière sécurisée"""
        try:
            if isinstance(content, (dict, list)):
                # Pour les dict/list, créer une représentation textuelle significative
                if isinstance(content, dict):
                    # Extraire les clés importantes
                    important_items = {}
                    for key, value in content.items():
                        if key in ['name', 'type', 'content', 'code', 'body', 'value']:
                            if isinstance(value, str):
                                important_items[key] = value
                            else:
                                important_items[key] = str(value)[:200]

                    if important_items:
                        return f"Chunk data: {important_items}"
                    else:
                        return f"Chunk data: {str(content)[:500]}"

                elif isinstance(content, list):
                    # Pour les listes, prendre les premiers éléments
                    preview = [str(item)[:100] for item in content[:3]]
                    return f"Chunk list: {preview}"

            # Conversion générique
            return str(content)[:1000]

        except Exception as e:
            logger.warning(f"Erreur conversion contenu: {e}")
            return "Content conversion error"

    def _extract_content_from_dict(self, chunk_dict: Dict) -> Optional[str]:
        """Extrait le contenu textuel d'un dictionnaire de chunk"""
        # Priorité des clés pour le contenu
        content_keys = ['content', 'text', 'code', 'value', 'data', 'body']

        for key in content_keys:
            if key in chunk_dict and chunk_dict[key]:
                content = chunk_dict[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, (dict, list)):
                    return str(content)[:1000]  # Limiter la taille

        # Essayer de convertir tout le dict en string significative
        meaningful_keys = {k: v for k, v in chunk_dict.items()
                           if not k.startswith('_') and not callable(v)}
        if meaningful_keys:
            return str(meaningful_keys)[:800]

        return None

    # src/code_vectorizer.py (extrait - méthode _create_standard_chunk)
    def _create_standard_chunk(self, content: str, chunk_type: str,
                               file_info: Dict[str, Any], index: int,
                               original_chunk: Optional[Dict] = None) -> Dict[str, Any]:
        """Crée un chunk standardisé avec métadonnées ChromaDB-compatibles"""
        # S'assurer que le contenu est une string
        if not isinstance(content, str):
            content = self._convert_to_string(content)

        # Métadonnées de base (types primitifs uniquement)
        metadata = {
            'file_path': file_info['path'],
            'relative_path': file_info['relative_path'],
            'filename': file_info['filename'],
            'extension': file_info['extension'],
            'chunk_index': index,
            'chunk_id': f"{Path(file_info['path']).stem}_{index}",
            'chunk_type': chunk_type
        }

        # Ajouter les métadonnées originales si disponibles (sérialisées)
        if original_chunk and isinstance(original_chunk, dict):
            for key in ['line_start', 'line_end', 'language', 'category']:
                if key in original_chunk and original_chunk[key] is not None:
                    value = original_chunk[key]
                    # Convertir en types ChromaDB-compatibles
                    if isinstance(value, (list, dict)):
                        metadata[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        metadata[key] = value

        chunk = {
            'content': content[:2000],  # Limiter la taille
            'type': chunk_type,
            'metadata': metadata
        }

        return chunk

    def _enrich_chunk_metadata(self, chunk: Dict[str, Any], file_info: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Enrichit les métadonnées d'un chunk existant"""
        if 'metadata' not in chunk:
            chunk['metadata'] = {}

        # Métadonnées de base
        chunk['metadata'].update({
            'file_path': file_info['path'],
            'relative_path': file_info['relative_path'],
            'filename': file_info['filename'],
            'extension': file_info['extension'],
            'chunk_index': index,
            'chunk_id': f"{Path(file_info['path']).stem}_{index}"
        })

        return chunk

    def _vectorize_and_store_chunks(self, file_info: Dict[str, Any],
                                    chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vectorise et stocke les chunks normalisés"""
        try:
            # Extraire le contenu pour la vectorisation
            chunk_contents = [chunk['content'] for chunk in chunks]

            # Vectorisation par lots pour les gros fichiers
            batch_size = self.vectorization_config.get('batch_size', 50)
            all_embeddings = []

            for i in range(0, len(chunk_contents), batch_size):
                batch = chunk_contents[i:i + batch_size]
                batch_embeddings = self.model.encode(batch)
                all_embeddings.extend(batch_embeddings)

            # Convertir en numpy array
            embeddings = np.array(all_embeddings)

            # Stockage dans la base vectorielle
            self.vector_store.add_chunks(file_info, chunks, embeddings)

            logger.debug(f"✅ {len(chunks)} chunks vectorisés pour {file_info['filename']}")
            return chunks

        except Exception as e:
            logger.error(f"Erreur vectorisation pour {file_info['path']}: {e}")
            return None

    def _fallback_file_processing(self, file_info: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Traitement de fallback pour les fichiers problématiques"""
        try:
            with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return None

            # Créer un chunk simple avec le contenu brut
            chunk = self._create_standard_chunk(
                content[:1500],  # Limiter la taille
                'raw_fallback',
                file_info,
                0
            )

            # Vectoriser et stocker le chunk de fallback
            return self._vectorize_and_store_chunks(file_info, [chunk])

        except Exception as e:
            logger.error(f"Fallback échoué pour {file_info['path']}: {e}")
            return None

    def _log_progress(self, stats: Dict[str, Any]):
        """Log la progression actuelle"""
        progress = (stats['processed_files'] / stats['total_files']) * 100
        logger.info(
            f"📊 Progression: {stats['processed_files']}/{stats['total_files']} "
            f"fichiers ({progress:.1f}%) - {stats['total_chunks']} chunks créés"
        )

    def _log_final_stats(self, stats: Dict[str, Any]):
        """Log les statistiques finales"""
        logger.info("=" * 60)
        logger.info("✅ VECTORISATION TERMINÉE")
        logger.info("=" * 60)
        logger.info(f"📁 Fichiers totaux: {stats['total_files']}")
        logger.info(f"✅ Fichiers traités: {stats['processed_files']}")
        logger.info(f"❌ Fichiers échoués: {stats['failed_files']}")
        logger.info(f"🪓 Chunks créés: {stats['total_chunks']}")

        if stats['files_by_extension']:
            logger.info("📊 Fichiers par extension:")
            for ext, count in sorted(stats['files_by_extension'].items(),
                                     key=lambda x: x[1], reverse=True):
                logger.info(f"  {ext}: {count}")

        if stats['chunks_by_type']:
            logger.info("🎯 Chunks par type:")
            for chunk_type, count in sorted(stats['chunks_by_type'].items(),
                                            key=lambda x: x[1], reverse=True):
                logger.info(f"  {chunk_type}: {count}")

    def search_similar_code(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche du code similaire à la requête"""
        try:
            query_embedding = self.model.encode([query])
            results = self.vector_store.search(query_embedding, top_k)
            logger.debug(f"🔍 Recherche: '{query}' → {len(results)} résultats")
            return results
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la base vectorielle"""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"Erreur récupération statistiques: {e}")
            return {'error': str(e)}

    def cleanup(self):
        """Nettoie les ressources"""
        if hasattr(self, 'vector_store'):
            self.vector_store.save()
        logger.info("🧹 CodeVectorizer nettoyé")