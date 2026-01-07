import json
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import time
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from file.file_info import FileInfo
from file.file_scanner import FileScanner
from parsers.multilanguage_analyzer import MultilanguageAnalyzer
from vector.vector_model import FileStatus, VectorizationStats, ChunkType, FileProcessingResult, ChunkMetadata, \
    NormalizedChunk
from vector.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Types pour la compatibilité avec les API existantes
RawChunk = Union[Dict[str, Any], str, List[Any], tuple]
RawChunks = List[RawChunk]
NormalizedChunkDict = Dict[str, Any]
NormalizedChunksList = List[NormalizedChunkDict]

class CodeVectorizer:
    """Vectoriser de code qui transforme le code source en embeddings vectoriels"""

    def __init__(self, config: Dict[str, Any], model: SentenceTransformer):
        self.config = config
        self.analyzer = MultilanguageAnalyzer()
        self.vectorization_config = config.get('vectorization', {})

        # Initialisation des composants
        self.file_scanner = FileScanner(config.get('project', {}))
        self.vector_store = VectorStore(self.vectorization_config)

        # Modèle embedding
        self.model = model

        # Configuration des limites
        self.max_chunk_size = self.vectorization_config.get('max_chunk_size', 5000)
        self.min_chunk_size = self.vectorization_config.get('min_chunk_size', 50)
        self.batch_size = self.vectorization_config.get('batch_size', 50)

        logger.info("CodeVectorizer initialisé")

    def vectorize_project(self) -> VectorizationStats:
        """Vectorise l'ensemble du projet avec suivi de progression et gestion robuste"""
        stats = VectorizationStats(start_time=time.time())

        logger.info("🚀 Début de la vectorisation du projet...")

        try:
            for file_info in self.file_scanner.scan_project():
                stats.total_files += 1

                try:
                    result = self._process_file_safe(file_info, stats)

                    if result.success:
                        stats.processed_files += 1
                        stats.files_by_status[FileStatus.SUCCESS] = stats.files_by_status.get(FileStatus.SUCCESS, 0) + 1
                        stats.total_chunks += len(result.chunks)
                        self._update_file_stats(stats, file_info, result.chunks)
                    else:
                        stats.failed_files += 1
                        stats.files_by_status[FileStatus.FAILED] = stats.files_by_status.get(FileStatus.FAILED, 0) + 1
                        if result.fallback_used:
                            stats.fallback_files += 1
                            stats.files_by_status[FileStatus.FALLBACK] = stats.files_by_status.get(FileStatus.FALLBACK,
                                                                                                   0) + 1

                except Exception as e:
                    stats.failed_files += 1
                    stats.files_by_status[FileStatus.FAILED] = stats.files_by_status.get(FileStatus.FAILED, 0) + 1
                    logger.error(f"Erreur critique traitement fichier {file_info.path}: {e}")
                    logger.debug(traceback.format_exc())

                # Log de progression
                if stats.total_files % 50 == 0:
                    self._log_progress(stats)

        except Exception as e:
            logger.error(f"Erreur pendant le scan du projet: {e}")
            stats.scan_error = str(e)

        # Finalisation
        stats.end_time = time.time()
        stats.duration_seconds = stats.end_time - stats.start_time
        self._log_final_stats(stats)

        try:
            self.vector_store.persist_index()
            stats.persisted = True
        except Exception as e:
            logger.error(f"Erreur sauvegarde index: {e}")
            stats.persisted = False
            stats.persist_error = str(e)

        return stats

    def _process_file_safe(self, file_info: FileInfo, stats: VectorizationStats) -> FileProcessingResult:
        """Traite un fichier de manière sécurisée avec gestion d'erreurs complète"""
        result = FileProcessingResult(
            success=False,
            chunks=[],
            error=None,
            fallback_used=False,
            analysis_status=None
        )

        try:
            # 1. Analyse du fichier
            analysis_result = self.analyzer.analyze_file(file_info.path)

            # 2. Création des chunks - conversion de CodeChunk à RawChunk
            code_chunks = analysis_result.chunks
            chunks: RawChunks = self._convert_code_chunks_to_raw(code_chunks)

            if not chunks:
                logger.warning(f"Aucun chunk créé pour: {file_info.path}")
                return result

            # 3. Normalisation des chunks
            normalized_chunks = self._normalize_chunks(chunks, file_info)
            if not normalized_chunks:
                logger.warning(f"Aucun chunk normalisé pour: {file_info.path}")
                return result

            # 4. Validation des chunks
            validated_chunks = self._validate_chunks(normalized_chunks)
            if not validated_chunks:
                logger.warning(f"Aucun chunk valide pour: {file_info.path}")
                return result

            # 5. Conversion en NormalizedChunk
            typed_chunks = self._convert_to_normalized_chunks(validated_chunks)
            if not typed_chunks:
                logger.warning(f"Aucun chunk typé valide pour: {file_info.path}")
                return result

            # 6. Vectorisation et stockage
            vectorized_chunks = self._vectorize_and_store_chunks(file_info, typed_chunks)
            if vectorized_chunks:
                result.chunks = vectorized_chunks
                result.success = True
                stats.total_vector_size += len(vectorized_chunks)

        except Exception as e:
            print(traceback.format_exc())
            result.error = str(e)
            logger.error(f"Erreur traitement fichier {file_info.path}: {e}")
            logger.debug(traceback.format_exc())

            # Essayer le fallback en cas d'erreur
            try:
                fallback_chunks = self._fallback_file_processing(file_info)
                if fallback_chunks:
                    result.chunks = fallback_chunks
                    result.success = True
                    result.fallback_used = True
            except Exception as fallback_error:
                logger.error(f"Fallback aussi échoué pour {file_info.path}: {fallback_error}")

        return result

    def _convert_code_chunks_to_raw(self, code_chunks: List[Any]) -> RawChunks:
        """Convertit les CodeChunk en RawChunk pour la normalisation"""
        raw_chunks: RawChunks = []

        for chunk in code_chunks:
            # Si c'est déjà un CodeChunk, extraire les informations
            if hasattr(chunk, 'content') and hasattr(chunk, 'language'):
                # Créer un dictionnaire avec toutes les informations du CodeChunk
                chunk_dict = {
                    'content': chunk.content,
                    'language': chunk.language,
                    'name': chunk.name,
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'name_parts': getattr(chunk, 'name_parts', []),
                    'code_chunk': chunk  # Garder une référence au chunk original
                }
                raw_chunks.append(chunk_dict)
            elif isinstance(chunk, dict):
                # C'est déjà un dictionnaire
                raw_chunks.append(chunk)
            else:
                # Autre type, convertir en string
                raw_chunks.append(str(chunk))

        return raw_chunks

    def _normalize_chunks(self, chunks: RawChunks, file_info: FileInfo) -> NormalizedChunksList:
        """Normalise la structure des chunks pour assurer la cohérence"""
        normalized_chunks: NormalizedChunksList = []

        for i, chunk in enumerate(chunks):
            try:
                normalized_chunk = self._normalize_single_chunk(chunk, i, file_info)
                if normalized_chunk:
                    normalized_chunks.append(normalized_chunk)
            except Exception as e:
                logger.warning(f"Erreur normalisation chunk {i} pour {file_info.path}: {e}")

        return normalized_chunks

    def _normalize_single_chunk(self, chunk: RawChunk, index: int, file_info: FileInfo) -> Optional[
        NormalizedChunkDict]:
        """Normalise un chunk individuel"""
        try:
            normalized_chunk: Optional[NormalizedChunkDict] = None

            # Gestion des différents types de chunks
            if isinstance(chunk, dict):
                normalized_chunk = self._normalize_dict_chunk(chunk, file_info, index)
            elif isinstance(chunk, str):
                normalized_chunk = self._normalize_string_chunk(chunk, file_info, index)
            elif isinstance(chunk, (list, tuple)):
                normalized_chunk = self._normalize_list_chunk(chunk, file_info, index)
            else:
                normalized_chunk = self._normalize_other_chunk(chunk, file_info, index)

            # Post-normalisation
            if normalized_chunk:
                normalized_chunk = self._post_process_chunk(normalized_chunk, file_info, index)

            return normalized_chunk

        except Exception as e:
            logger.error(f"Erreur normalisation chunk: {e}")
            return None

    def _normalize_dict_chunk(self, chunk: Dict[str, Any], file_info: FileInfo, index: int) -> Optional[
        NormalizedChunkDict]:
        """Normalise un chunk de type dictionnaire"""
        # Si c'est déjà un chunk normalisé
        if 'content' in chunk and isinstance(chunk['content'], str):
            return self._enrich_chunk_metadata(chunk, file_info, index)

        # Essayer d'extraire du contenu
        content = self._extract_content_from_dict(chunk)
        if content:
            return self._create_standard_chunk(content, ChunkType.NORMALIZED_DICT, file_info, index, chunk)

        # Convertir tout le dict en JSON
        try:
            content = json.dumps(chunk, ensure_ascii=False, default=str)
            return self._create_standard_chunk(content, ChunkType.DICT_JSON, file_info, index)
        except Exception as e:
            logger.warning(f"Erreur conversion dict en JSON: {e}")
            # Fallback: représentation texte
            content = str(chunk)[:self.max_chunk_size]
            return self._create_standard_chunk(content, ChunkType.DICT_FALLBACK, file_info, index)

    def _normalize_string_chunk(self, chunk: str, file_info: FileInfo, index: int) -> NormalizedChunkDict:
        """Normalise un chunk de type string"""
        return self._create_standard_chunk(chunk, ChunkType.TEXT, file_info, index)

    def _normalize_list_chunk(self, chunk: List, file_info: FileInfo, index: int) -> Optional[NormalizedChunkDict]:
        """Normalise un chunk de type liste"""
        try:
            # Essayer de convertir en JSON
            content = json.dumps(chunk, ensure_ascii=False, default=str)
            return self._create_standard_chunk(content, ChunkType.LIST_JSON, file_info, index)
        except Exception as e:
            logger.warning(f"Erreur conversion liste en JSON: {e}")
            # Fallback : représentation texte
            content = str(chunk)[:self.max_chunk_size]
            return self._create_standard_chunk(content, ChunkType.LIST_FALLBACK, file_info, index)

    def _normalize_other_chunk(self, chunk: Any, file_info: FileInfo, index: int) -> NormalizedChunkDict:
        """Normalise d'autres types de chunks"""
        content = self._convert_to_string(chunk)
        return self._create_standard_chunk(content, ChunkType.OTHER, file_info, index)

    def _extract_content_from_dict(self, chunk_dict: Dict[str, Any]) -> Optional[str]:
        """Extrait le contenu textuel d'un dictionnaire de chunk"""
        # Priorité des clés pour le contenu
        content_keys = ['content', 'text', 'code', 'body', 'value', 'data',
                        'description', 'name', 'title', 'summary', 'definition']

        for key in content_keys:
            if key in chunk_dict and chunk_dict[key] is not None:
                value = chunk_dict[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (dict, list)):
                    try:
                        return json.dumps(value, ensure_ascii=False, default=str)[:self.max_chunk_size]
                    except:
                        return str(value)[:self.max_chunk_size]

        # Essayer de trouver n'importe quel contenu textuel
        for key, value in chunk_dict.items():
            if isinstance(value, str) and len(value) > self.min_chunk_size:
                return value

        return None

    def _convert_to_string(self, content: Any) -> str:
        """Convertit n'importe quel contenu en string de manière sécurisée"""
        try:
            if isinstance(content, dict):
                # Pour les dicts, créer une représentation structurée
                important_items: Dict[str, str] = {}
                for key, value in content.items():
                    if key and value is not None:
                        if isinstance(value, str):
                            if len(value) < 200:  # Éviter les très longs strings
                                important_items[key] = value
                        else:
                            try:
                                important_items[key] = json.dumps(value, ensure_ascii=False, default=str)[:100]
                            except:
                                important_items[key] = str(value)[:100]

                if important_items:
                    return f"Data: {important_items}"

            elif isinstance(content, list):
                # Pour les listes, montrer les premiers éléments
                preview: List[str] = []
                for i, item in enumerate(content[:5]):  # Limiter à 5 éléments
                    if isinstance(item, str):
                        preview.append(item[:50])
                    else:
                        preview.append(str(item)[:50])
                return f"List[{len(content)}]: {preview}"

            # Conversion générique
            result = str(content)
            return result[:self.max_chunk_size]

        except Exception as e:
            logger.warning(f"Erreur conversion contenu: {e}")
            return f"Content conversion error: {str(e)[:100]}"

    def _create_standard_chunk(self, content: str, chunk_type: ChunkType,
                               file_info: FileInfo, index: int,
                               original_chunk: Optional[Dict[str, Any]] = None) -> NormalizedChunkDict:
        """Crée un chunk standardisé avec métadonnées enrichies"""
        try:
            # Validation et nettoyage du contenu
            if not isinstance(content, str):
                content = self._convert_to_string(content)

            if not content or not content.strip():
                content = f"Empty content - {chunk_type.value}"

            # Limiter la taille
            if len(content) > self.max_chunk_size:
                content = content[:self.max_chunk_size] + "... [truncated]"

            # Métadonnées de base (compatibles avec ChromaDB)
            metadata_dict: Dict[str, Any] = {
                'file_path': file_info.path,
                'relative_path': file_info.relative_path,
                'filename': file_info.filename,
                'extension': file_info.extension or '',
                'chunk_index': index,
                'chunk_id': f"{Path(file_info.path).stem}_{index}_{hash(file_info.path) % 10000:04d}",
                'chunk_type': chunk_type.value,
                'chunk_size': len(content)
            }

            # Ajouter des métadonnées spécifiques si disponibles
            if original_chunk and isinstance(original_chunk, dict):
                self._add_original_metadata(metadata_dict, original_chunk)

            # Créer le chunk final
            chunk: NormalizedChunkDict = {
                'id': metadata_dict['chunk_id'],
                'content': content,
                'type': chunk_type.value,
                'metadata': metadata_dict,
                'embedding': None  # Sera rempli plus tard
            }

            return chunk

        except Exception as e:
            logger.error(f"Erreur création chunk standard: {e}")
            # Chunk minimal de secours
            return {
                'id': f"error_{index}_{hash(file_info.path) % 10000:04d}",
                'content': f"Error creating chunk: {str(e)[:200]}",
                'type': ChunkType.ERROR.value,
                'metadata': {
                    'file_path': file_info.path,
                    'chunk_index': index,
                    'error': str(e)[:100]
                },
                'embedding': None
            }

    def _add_original_metadata(self, metadata: Dict[str, Any], original_chunk: Dict[str, Any]):
        """Ajoute les métadonnées originales au chunk"""
        # Clés à copier directement (si elles sont de types primitifs)
        primitive_keys = ['line_start', 'line_end', 'language', 'category',
                          'function_name', 'class_name', 'method_name',
                          'element_type', 'modifier', 'visibility']

        for key in primitive_keys:
            if key in original_chunk:
                value = original_chunk[key]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value

        # Clés à sérialiser si nécessaire
        complex_keys = ['parameters', 'modifiers', 'attributes', 'metadata', 'dependencies']
        for key in complex_keys:
            if key in original_chunk:
                value = original_chunk[key]
                if isinstance(value, (dict, list)):
                    try:
                        metadata[key] = json.dumps(value, ensure_ascii=False, default=str)[:500]
                    except:
                        metadata[key] = str(value)[:500]
                elif isinstance(value, str):
                    metadata[key] = value[:500]

    def _enrich_chunk_metadata(self, chunk: NormalizedChunkDict, file_info: FileInfo,
                               index: int) -> NormalizedChunkDict:
        """Enrichit les métadonnées d'un chunk existant"""
        if 'metadata' not in chunk:
            chunk['metadata'] = {}

        # Métadonnées de base
        chunk['metadata'].update({
            'file_path': file_info.path,
            'relative_path': file_info.relative_path,
            'filename': file_info.filename,
            'extension': file_info.extension or '',
            'chunk_index': index,
            'chunk_id': chunk.get('metadata', {}).get('chunk_id', f"{Path(file_info.path).stem}_{index}")
        })

        # S'assurer que l'ID est présent
        if 'id' not in chunk:
            chunk['id'] = chunk['metadata']['chunk_id']

        return chunk

    def _post_process_chunk(self, chunk: NormalizedChunkDict, file_info: FileInfo, index: int) -> NormalizedChunkDict:
        """Post-traitement du chunk normalisé"""
        # S'assurer que le contenu n'est pas vide
        if not chunk.get('content', '').strip():
            chunk['content'] = f"Processed chunk from {file_info.filename} (index: {index})"

        # S'assurer que l'ID est unique
        if 'id' not in chunk:
            chunk['id'] = f"{Path(file_info.path).stem}_{index}_{hash(file_info.path) % 10000:04d}"

        # Ajouter un timestamp
        if 'metadata' in chunk:
            chunk['metadata']['processed_at'] = time.time()

        return chunk

    def _validate_chunks(self, chunks: NormalizedChunksList) -> NormalizedChunksList:
        """Valide et filtre les chunks"""
        valid_chunks: NormalizedChunksList = []

        for chunk in chunks:
            try:
                # Validation de base
                if not isinstance(chunk, dict):
                    continue

                if 'content' not in chunk or not isinstance(chunk['content'], str):
                    continue

                if not chunk['content'].strip():
                    continue

                # Validation de la taille
                content_len = len(chunk['content'])
                if content_len < self.min_chunk_size:
                    logger.debug(f"Chunk trop petit ({content_len} chars), ignoré")
                    continue

                if content_len > self.max_chunk_size * 2:  # Un peu plus permissif
                    logger.warning(f"Chunk très grand ({content_len} chars), tronqué")
                    chunk['content'] = chunk['content'][:self.max_chunk_size] + "... [truncated]"

                # S'assurer que les métadonnées existent
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}

                if 'id' not in chunk:
                    chunk['id'] = chunk.get('metadata', {}).get('chunk_id', f"chunk_{len(valid_chunks)}")

                valid_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Erreur validation chunk: {e}")
                continue

        return valid_chunks

    def _convert_to_normalized_chunks(self, chunks: NormalizedChunksList) -> List[NormalizedChunk]:
        """Convertit les chunks dict en objets NormalizedChunk typés"""
        typed_chunks: List[NormalizedChunk] = []

        for chunk_dict in chunks:
            try:
                # Créer l'objet ChunkMetadata
                metadata_dict = chunk_dict.get('metadata', {})
                metadata = ChunkMetadata(
                    file_path=metadata_dict.get('file_path', ''),
                    relative_path=metadata_dict.get('relative_path', ''),
                    filename=metadata_dict.get('filename', ''),
                    extension=metadata_dict.get('extension', ''),
                    chunk_index=metadata_dict.get('chunk_index', 0),
                    chunk_id=metadata_dict.get('chunk_id', chunk_dict.get('id', '')),
                    chunk_type=ChunkType(metadata_dict.get('chunk_type', ChunkType.TEXT.value)),
                    chunk_size=metadata_dict.get('chunk_size', len(chunk_dict.get('content', ''))),
                    line_start=metadata_dict.get('line_start'),
                    line_end=metadata_dict.get('line_end'),
                    language=metadata_dict.get('language'),
                    category=metadata_dict.get('category'),
                    function_name=metadata_dict.get('function_name'),
                    class_name=metadata_dict.get('class_name'),
                    method_name=metadata_dict.get('method_name'),
                    element_type=metadata_dict.get('element_type'),
                    modifier=metadata_dict.get('modifier'),
                    visibility=metadata_dict.get('visibility'),
                    parameters=metadata_dict.get('parameters'),
                    modifiers=metadata_dict.get('modifiers'),
                    attributes=metadata_dict.get('attributes'),
                    dependencies=metadata_dict.get('dependencies'),
                    processed_at=metadata_dict.get('processed_at', time.time()),
                    error=metadata_dict.get('error')
                )

                typed_chunk = NormalizedChunk(
                    id=chunk_dict.get('id', metadata.chunk_id),
                    content=chunk_dict.get('content', ''),
                    type=ChunkType(chunk_dict.get('type', ChunkType.TEXT.value)),
                    metadata=metadata,
                    embedding=chunk_dict.get('embedding')
                )
                typed_chunks.append(typed_chunk)

            except Exception as e:
                logger.warning(f"Erreur conversion chunk typé: {e}")
                continue

        return typed_chunks

    def _vectorize_and_store_chunks(self, file_info: FileInfo,
                                    chunks: List[NormalizedChunk]) -> Optional[List[NormalizedChunk]]:
        """Vectorise et stocke les chunks normalisés (version compatible)"""
        try:
            if not chunks:
                return None

            # Extraire le contenu pour la vectorisation
            chunk_contents: List[str] = []
            valid_chunks: List[NormalizedChunk] = []

            for chunk in chunks:
                content = chunk.content
                if content and isinstance(content, str) and content.strip():
                    chunk_contents.append(content)
                    valid_chunks.append(chunk)
                else:
                    logger.debug(f"Chunk ignoré (contenu vide) pour {file_info.path}")

            if not valid_chunks:
                return None

            # Vectorisation par lots
            all_embeddings: List[NDArray] = []
            embedding_errors = 0

            for i in range(0, len(chunk_contents), self.batch_size):
                batch = chunk_contents[i:i + self.batch_size]
                try:
                    batch_embeddings = self.vector_encode(batch)
                    all_embeddings.append(batch_embeddings)
                except Exception as batch_error:
                    logger.error(f"Erreur vectorisation lot {i}: {batch_error}")
                    embedding_errors += 1

                    # Créer des embeddings de secours
                    embedding_dim = self.model.get_sentence_embedding_dimension()
                    dummy_embedding = np.zeros((len(batch), embedding_dim))
                    all_embeddings.append(dummy_embedding)

            # Combiner tous les embeddings
            if all_embeddings:
                try:
                    embeddings = np.vstack(all_embeddings)
                except Exception as e:
                    logger.error(f"Erreur combinaison embeddings: {e}")
                    # Créer des embeddings nuls
                    embedding_dim = self.model.get_sentence_embedding_dimension()
                    embeddings = np.zeros((len(valid_chunks), embedding_dim))
            else:
                # Aucun embedding créé
                embedding_dim = self.model.get_sentence_embedding_dimension()
                embeddings = np.zeros((len(valid_chunks), embedding_dim))

            # Associer les embeddings aux chunks
            for i, chunk in enumerate(valid_chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i].tolist()
                else:
                    # Embedding de secours
                    chunk.embedding = [0.0] * embedding_dim

            # Stockage dans la base vectorielle
            try:
                # Préparer les chunks pour l'interface compatible
                prepared_chunks: NormalizedChunksList = []
                for chunk in valid_chunks:
                    prepared_chunk: NormalizedChunkDict = {
                        'content': chunk.content,
                        'metadata': {
                            'file_path': chunk.metadata.file_path,
                            'relative_path': chunk.metadata.relative_path,
                            'filename': chunk.metadata.filename,
                            'extension': chunk.metadata.extension,
                            'chunk_index': chunk.metadata.chunk_index,
                            'chunk_id': chunk.metadata.chunk_id,
                            'chunk_type': chunk.metadata.chunk_type.value,
                            'chunk_size': chunk.metadata.chunk_size,
                        },
                        'embedding': chunk.embedding,
                        'id': chunk.id
                    }
                    # Ajouter les métadonnées supplémentaires si présentes
                    for key in ['language', 'line_start', 'line_end', 'function_name', 'class_name']:
                        value = getattr(chunk.metadata, key, None)
                        if value is not None:
                            prepared_chunk['metadata'][key] = value

                    prepared_chunks.append(prepared_chunk)

                # Appel compatible avec la nouvelle interface
                self.vector_store.add_chunks(file_info.path, prepared_chunks, embeddings)

                logger.debug(f"✅ {len(valid_chunks)} chunks vectorisés pour {file_info.filename}")

                if embedding_errors > 0:
                    logger.warning(f"{embedding_errors} erreurs embedding pour {file_info.filename}")

                return valid_chunks

            except Exception as store_error:
                logger.error(f"Erreur stockage chunks pour {file_info.path}: {store_error}")
                return None

        except Exception as e:
            logger.error(f"Erreur vectorisation pour {file_info.path}: {e}", exc_info=True)
            return None

    def vector_encode(self, batch: List[str]) -> NDArray:
        """Encode un batch de textes en embeddings"""
        batch_embeddings = self.model.encode(
            batch,
            batch_size=min(len(batch), 32),  # Limiter la taille du batch
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return batch_embeddings

    def _fallback_file_processing(self, file_info: FileInfo) -> Optional[List[NormalizedChunk]]:
        """Traitement de fallback pour les fichiers problématiques"""
        try:
            # Essayer différents encodages
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            content = ""

            for encoding in encodings:
                try:
                    with open(file_info.path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()

                    if content.strip():
                        break
                except:
                    continue
            else:
                # Tous les encodings ont échoué
                logger.warning(f"Impossible de lire {file_info.path} avec les encodings disponibles")
                return None

            if not content or not content.strip():
                return None

            # Créer un chunk simple avec le contenu brut
            chunk_dict = self._create_standard_chunk(
                content[:self.max_chunk_size],
                ChunkType.RAW_FALLBACK,
                file_info,
                0
            )

            # Convertir en NormalizedChunk
            typed_chunks = self._convert_to_normalized_chunks([chunk_dict])
            if not typed_chunks:
                return None

            # Vectoriser et stocker le chunk de fallback
            return self._vectorize_and_store_chunks(file_info, typed_chunks)

        except Exception as e:
            logger.error(f"Fallback échoué pour {file_info.path}: {e}")
            return None

    def _update_file_stats(self, stats: VectorizationStats, file_info: FileInfo, chunks: List[NormalizedChunk]):
        """Met à jour les statistiques pour un fichier"""
        # Statistiques par extension
        extension = file_info.extension or 'unknown'
        stats.files_by_extension[extension] = stats.files_by_extension.get(extension, 0) + 1

        # Statistiques par type de chunk
        for chunk in chunks:
            try:
                chunk_type = chunk.type
                stats.chunks_by_type[chunk_type] = stats.chunks_by_type.get(chunk_type, 0) + 1
            except (ValueError, AttributeError):
                # Type inconnu, utiliser OTHER
                stats.chunks_by_type[ChunkType.OTHER] = stats.chunks_by_type.get(ChunkType.OTHER, 0) + 1

    def _log_progress(self, stats: VectorizationStats):
        """Log la progression actuelle"""
        if stats.total_files > 0:
            progress = (stats.processed_files / stats.total_files) * 100
            logger.info(
                f"📊 Progression: {stats.processed_files}/{stats.total_files} "
                f"fichiers ({progress:.1f}%) - {stats.total_chunks} chunks créés"
            )

            # Afficher les erreurs
            if stats.failed_files > 0:
                logger.warning(f"   ⚠️  {stats.failed_files} fichiers échoués")

            if stats.fallback_files > 0:
                logger.warning(f"   🔄 {stats.fallback_files} fichiers en fallback")

    def _log_final_stats(self, stats: VectorizationStats):
        """Log les statistiques finales"""
        logger.info("=" * 60)
        logger.info("✅ VECTORISATION TERMINÉE")
        logger.info("=" * 60)
        logger.info(f"📁 Fichiers totaux: {stats.total_files}")
        logger.info(f"✅ Fichiers traités: {stats.processed_files}")
        logger.info(f"❌ Fichiers échoués: {stats.failed_files}")

        if stats.fallback_files > 0:
            logger.info(f"🔄 Fichiers fallback: {stats.fallback_files}")

        logger.info(f"🪓 Chunks créés: {stats.total_chunks}")

        if stats.total_vector_size > 0:
            logger.info(f"🧮 Taille vectorielle totale: {stats.total_vector_size} embeddings")

        # Statistiques par extension
        if stats.files_by_extension:
            logger.info("\n📊 Fichiers par extension (top 10):")
            sorted_extensions = sorted(stats.files_by_extension.items(),
                                       key=lambda x: x[1], reverse=True)[:10]
            for ext, count in sorted_extensions:
                percentage = (count / stats.total_files) * 100 if stats.total_files > 0 else 0
                logger.info(f"  {ext:15} {count:4} ({percentage:.1f}%)")

        # Statistiques par type de chunk
        if stats.chunks_by_type and stats.total_chunks > 0:
            logger.info("\n🎯 Chunks par type (top 10):")
            sorted_chunk_types = sorted(stats.chunks_by_type.items(),
                                        key=lambda x: x[1], reverse=True)[:10]
            for chunk_type, count in sorted_chunk_types:
                percentage = (count / stats.total_chunks) * 100
                logger.info(f"  {chunk_type.value:20} {count:4} ({percentage:.1f}%)")

    # Méthodes de recherche et autres méthodes publiques
    def search_similar_code(self, query: str, top_k: int = 5,
                            threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Recherche du code similaire à la requête"""
        try:
            query = query.strip()
            if not query:
                return []

            query_embedding = self.model.encode([query], convert_to_numpy=True)
            results = self.vector_store.search(
                query_embedding[0],
                top_k=top_k,
                threshold=threshold
            )

            logger.debug(f"🔍 Recherche: '{query[:50]}...' → {len(results)} résultats")
            return results

        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return []

    def search_by_metadata(self, filters: Dict[str, Any],
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Recherche des chunks par métadonnées"""
        try:
            return self.vector_store.search_by_metadata(filters, limit)
        except Exception as e:
            logger.error(f"Erreur recherche par métadonnées: {e}")
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
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.persist_index()
                self.vector_store.cleanup()
            logger.info("🧹 CodeVectorizer nettoyé")
        except Exception as e:
            logger.error(f"Erreur nettoyage CodeVectorizer: {e}")

    def export_chunks(self, output_path: str, limit: Optional[int] = None) -> bool:
        """Exporte les chunks vers un fichier JSON"""
        try:
            chunks = self.vector_store.get_all_chunks(limit)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"✅ Chunks exportés vers {output_path} ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"Erreur export chunks: {e}")
            return False

    def get_chunk_debug_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère des informations de débogage pour un chunk spécifique"""
        try:
            return self.vector_store.get_chunk_by_id(chunk_id)
        except Exception as e:
            logger.error(f"Erreur récupération chunk {chunk_id}: {e}")
            return None
