# src/vector/vector_store.py
import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
import numpy as np
import time
from chromadb import Settings

from models.code_chunk import CodeChunk
from models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get("database", {})
        self.client = None
        self.collection = None
        self._index = defaultdict(dict)
        self._embedding_function = None
        self._initialize_database()

    def _initialize_database(self):
        persist_path = self.db_config.get("path", "./vector_db")
        collection_name = self.db_config.get("collection_name", "code_chunks")

        settings = Settings()
        settings.anonymized_telemetry = False
        self.client = chromadb.PersistentClient(path=persist_path, settings=settings)

        self._setup_embedding_function()

        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self._embedding_function
            )
            logger.info(f"✅ Collection '{collection_name}' chargée ({self.collection.count()} chunks)")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self._embedding_function,
                metadata={
                    "description": "Code chunks embeddings",
                    "created_at": datetime.now().isoformat(),
                    "version": "2.0",
                },
            )
            logger.info(f"✅ Nouvelle collection '{collection_name}' créée")

        self._load_index()

    def _load_index(self):
        """
        Charge l'index ChromaDB depuis le disque.
        Si le fichier n'existe pas, initialise un index vide.
        """
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"

            if index_path.exists():
                with open(index_path, 'rb') as f:
                    loaded_index = pickle.load(f)

                # Restaurer la structure
                for key in ['extension', 'chunk_type', 'file_path', 'language']:
                    if key in loaded_index:
                        self._index[key] = defaultdict(list, loaded_index[key])
                    else:
                        self._index[key] = defaultdict(list)

                logger.info(f"✅ Index chargé depuis {index_path} "
                            f"({len(self._index['extension'])} extensions, "
                            f"{len(self._index['chunk_type'])} types)")
            else:
                logger.info(f"⚠️ Index non trouvé à {index_path}, création d'un index vide")
                self._init_index_structure()

        except Exception as e:
            logger.error(f"❌ Impossible de charger l'index: {e}")
            self._init_index_structure()

    def persist_index(self):
        """Sauvegarde l'index sur disque (public)"""
        try:
            persist_path = self.db_config.get('path', './vector_db')
            index_path = Path(persist_path) / "index.pkl"
            index_path.parent.mkdir(parents=True, exist_ok=True)

            serializable_index = {k: dict(v) for k, v in self._index.items()}

            with open(index_path, 'wb') as f:
                pickle.dump(serializable_index, f)

            logger.info("💾 Index sauvegardé")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde index: {e}")

    def _init_index_structure(self):
        """Initialise la structure vide de l'index pour ChromaDB"""
        from collections import defaultdict

        self._index = {
            'extension': defaultdict(list),
            'chunk_type': defaultdict(list),
            'file_path': defaultdict(list),
            'language': defaultdict(list),
        }

    def _setup_embedding_function(self):
        try:
            from chromadb.utils import embedding_functions
            model_name = self.db_config.get("embedding_model", "all-MiniLM-L6-v2")
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception:
            self._embedding_function = None
            logger.warning("⚠️ Embedding par défaut utilisé")

    def get_all_chunks(self, limit: int = 2000) -> List[CodeChunk]:
        """Retourne tous les chunks sous forme typée CodeChunk"""
        try:
            results = self.collection.get(limit=limit, include=["documents", "metadatas"])
            chunks = []
            for i in range(len(results["documents"])):
                raw = {
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "id": results["ids"][i],
                }
                chunks.append(ModelFactory.create_code_chunk(raw))
            return chunks
        except Exception as e:
            logger.error(f"❌ Erreur récupération chunks: {e}")
            return []

    def search_chunks(self, query: str = None, embedding: np.ndarray = None, top_k: int = 5) -> List[CodeChunk]:
        """Recherche par texte ou embedding, retourne CodeChunk"""
        try:
            if embedding is not None:
                results = self.collection.query(
                    query_embeddings=[embedding.tolist()], n_results=top_k, include=["documents", "metadatas", "ids"]
                )
            else:
                results = self.collection.query(
                    query_texts=[query], n_results=top_k, include=["documents", "metadatas", "ids"]
                )
            chunks = []
            for i in range(len(results["documents"][0])):
                raw = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                }
                chunks.append(ModelFactory.create_code_chunk(raw))
            return chunks
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []

    def add_chunks(self, file_info: Dict[str, Any], chunks: List[Dict[str, Any]],
                   embeddings: Optional[np.ndarray] = None) -> List[str]:
        """
        Ajoute des chunks à la collection ChromaDB.

        :param file_info: infos sur le fichier, exemple: {'path': 'src/main.py', 'filename': 'main.py', 'extension': 'py'}
        :param chunks: liste de chunks, chaque chunk est un dict avec au moins 'content' et éventuellement 'metadata'
        :param embeddings: embeddings pré-calculés (optionnel)
        :return: liste des IDs ajoutés
        """
        ids = []
        try:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{hash(file_info.get('path', ''))}_{i}_{int(time.time() * 1000)}"
                content = chunk.get('content', '')[:10000]  # limiter taille

                metadata = chunk.get('metadata', {})
                # Au moment de créer enriched_metadata
                raw_metadata = {
                    'file_path': file_info.get('path', ''),
                    'filename': file_info.get('filename', ''),
                    'extension': file_info.get('extension', ''),
                    'chunk_index': i,
                    **metadata  # ce qui vient du chunk
                }

                # Nettoyer pour ChromaDB
                enriched_metadata = self._clean_metadata(raw_metadata)

                logger.info(f"📌 Chunk ready to add file: {file_info.get('filename')}, "
                            f"path: {file_info.get('path')}, chunk_type: {enriched_metadata.get('chunk_type')}")

                # Ajouter à ChromaDB
                if self._embedding_function:
                    self.collection.add(
                        documents=[content],
                        metadatas=[enriched_metadata],
                        ids=[chunk_id]
                    )
                else:
                    # embeddings fournis
                    emb = embeddings[i].tolist() if embeddings is not None else None
                    self.collection.add(
                        documents=[content],
                        embeddings=[emb] if emb is not None else None,
                        metadatas=[enriched_metadata],
                        ids=[chunk_id]
                    )

                ids.append(chunk_id)

                # Mettre à jour l'index interne
                self._index['extension'][enriched_metadata.get('extension', 'unknown')].append(chunk_id)
                self._index['file_path'][enriched_metadata.get('file_path', '')].append(chunk_id)
                self._index['language'][enriched_metadata.get('language', 'unknown')].append(chunk_id)
                self._index['chunk_type'][enriched_metadata.get('chunk_type', 'unknown')].append(chunk_id)

            self.persist_index()
            return ids

        except Exception as e:
            logger.error(f"❌ Erreur ajout chunks: {e}", exc_info=True)
            return []

    @staticmethod
    def _clean_metadata(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}

        for key, value in metadata_dict.items():
            if value is None:
                # Valeurs par défaut strictes pour ChromaDB
                cleaned[key] = ""  # on choisit str vide pour None
            elif isinstance(value, list) or isinstance(value, dict):
                # Sérialiser en string
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except:
                    cleaned[key] = str(value)
            elif isinstance(value, (int, float, bool, str)):
                cleaned[key] = value
            else:
                # Tout le reste en string
                cleaned[key] = str(value)
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales de la base vectorielle et de l'index"""
        try:
            return {
                'total_chunks': self.collection.count(),
                'collection_name': getattr(self.collection, 'name', None),
                'collection_metadata': getattr(self.collection, 'metadata', None),
                'embedding_function': 'custom' if self._embedding_function else 'default',
                'persist_path': self.db_config.get('path', './vector_db'),
                'index_stats': self.get_index_stats()
            }
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
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
                top_items = sorted(index_dict.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                stats[f'{key}_top_5'] = [{'value': val, 'count': len(ids)} for val, ids in top_items]
                return stats
    except Exception as e:
        (
            logger.error(f"❌ Erreur statistiques index: {e}")
        )
    return {'error': str(e)}
