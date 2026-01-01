# src/vector/vector_store.py
import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np
from chromadb import Settings

from models.code_chunk import CodeChunk
from parsers.utils.Util import infer_language_from_path, infer_category_from_type

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

    def create_code_chunk(self, raw_chunk: Dict[str, Any]) -> CodeChunk:
        metadata = raw_chunk.get("metadata", {})
        return CodeChunk(
            content=raw_chunk.get("content", ""),
            file_path=metadata.get("file_path", ""),
            filename=metadata.get("filename", ""),
            language=infer_language_from_path(metadata.get("file_path", "")),
            category=infer_category_from_type(metadata.get("chunk_type", ""), metadata.get("file_path", "")),
            chunk_type=metadata.get("chunk_type", "unknown"),
        )

    def get_all_chunks(self, limit: int = 2000) -> List[CodeChunk]:
        """Retourne tous les chunks sous forme typée CodeChunk"""
        try:
            results = self.collection.get(limit=limit, include=["documents", "metadatas", "ids"])
            chunks = []
            for i in range(len(results["documents"])):
                raw = {
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "id": results["ids"][i],
                }
                chunks.append(self.create_code_chunk(raw))
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
                chunks.append(self.create_code_chunk(raw))
            return chunks
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []
