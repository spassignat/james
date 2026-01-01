# src/vector_store.py
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

import chromadb
import numpy as np
from chromadb import Settings

from config.config_loader import ConfigLoader
from models.code_chunk import CodeChunk
from parsers.utils.Util import infer_language_from_path, infer_category_from_type, _auto_detect_features

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Gestion de la base vectorielle pour les CodeChunk
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get('database', {})
        self.client = None
        self.collection = None
        self._embedding_function = None
        self._cache = {}

        self._index = {
            'extension': defaultdict(list),
            'chunk_type': defaultdict(list),
            'file_path': defaultdict(list),
            'language': defaultdict(list)
        }

        self._initialize_database()

    def _initialize_database(self):
        """Initialise ChromaDB et l'index"""
        persist_path = self.db_config.get('path', './vector_db')
        collection_name = self.db_config.get('collection_name', 'code_chunks')

        settings = Settings()
        settings.anonymized_telemetry = False
        self.client = chromadb.PersistentClient(path=persist_path, settings=settings)

        self._setup_embedding_function()

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            )
            logger.info(f"✅ Collection '{collection_name}' chargée ({self.collection.count()} chunks)")
        except Exception:
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

        self._load_index()

    def _setup_embedding_function(self):
        try:
            from chromadb.utils import embedding_functions
            model_name = self.db_config.get('embedding_model', 'all-MiniLM-L6-v2')
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            logger.info(f"✅ Fonction d'embedding chargée: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️  Erreur chargement embedding: {e}, embedding par défaut")
            self._embedding_function = None

    def _load_index(self):
        """Charge l'index depuis le disque"""
        persist_path = Path(self.db_config.get('path', './vector_db')) / "index.pkl"
        if persist_path.exists():
            with open(persist_path, 'rb') as f:
                loaded_index = pickle.load(f)
            for key in self._index.keys():
                self._index[key] = defaultdict(list, loaded_index.get(key, {}))
            logger.info(f"✅ Index chargé ({len(self._index['extension'])} extensions)")
        else:
            logger.info("📊 Construction de l'index...")
            self._build_index()

    def _build_index(self, batch_size: int = 1000):
        """Construit l'index depuis la collection"""
        total = self.collection.count()
        if total == 0:
            return
        self._index = {k: defaultdict(list) for k in self._index.keys()}

        for offset in range(0, total, batch_size):
            batch = self.collection.get(limit=batch_size, offset=offset, include=["metadatas", "ids"])
            for i, metadata in enumerate(batch.get("metadatas", [])):
                chunk_id = batch["ids"][i]
                self._index['extension'][metadata.get('extension', 'unknown')].append(chunk_id)
                self._index['chunk_type'][metadata.get('chunk_type', 'unknown')].append(chunk_id)
                self._index['file_path'][metadata.get('file_path', '')].append(chunk_id)
                self._index['language'][metadata.get('language', 'unknown')].append(chunk_id)
        self._save_index()

    def _save_index(self):
        persist_path = Path(self.db_config.get('path', './vector_db')) / "index.pkl"
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_index = {k: dict(v) for k, v in self._index.items()}
        with open(persist_path, 'wb') as f:
            pickle.dump(serializable_index, f)
        logger.debug("💾 Index sauvegardé")

    def create_code_chunk(self, raw_chunk: Dict[str, Any]) -> CodeChunk:
        metadata = raw_chunk.get("metadata", {})
        return CodeChunk(
            content=raw_chunk["content"],
            file_path=metadata.get("file_path", ""),
            filename=metadata.get("filename", ""),
            language=infer_language_from_path(metadata.get("file_path", "")),
            category=infer_category_from_type(metadata.get("chunk_type", ""), metadata.get("file_path", "")),
            chunk_type=metadata.get("chunk_type", "unknown")
        )

    def get_chunks_for_agent(self, language: Optional[str] = None, category: Optional[str] = None) -> List[CodeChunk]:
        filters = {}
        if language:
            filters["language"] = language
        if category:
            filters["category"] = category
        results = self.search_by_metadata(filters, limit=200)
        return [self.create_code_chunk(r) for r in results]

    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Recherche dans la collection par métadonnées"""
        chunk_ids = []
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            if key in self._index and value in self._index[key]:
                chunk_ids = self._index[key][value][:limit]
        if chunk_ids:
            results = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
            return self._format_get_results(results)
        else:
            results = self.collection.get(where=filters, limit=limit, include=["documents", "metadatas"])
            return self._format_get_results(results)

    def _format_get_results(self, results: Dict) -> List[Dict[str, Any]]:
        formatted = []
        for i in range(len(results.get("documents", []))):
            formatted.append({
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "id": results["ids"][i]
            })
        return formatted

    def add_chunks(self, file_info: Dict[str, Any], chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Ajoute une liste de chunks avec embeddings et update index"""
        documents, metadatas, ids = [], [], []
        file_hash = hashlib.md5(file_info['path'].encode()).hexdigest()[:8]

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_hash}_{i}_{int(time.time()*1000)}"
            content = str(chunk.get("content", "")[:10000])
            documents.append(content)

            metadata = chunk.get("metadata", {})
            enriched_metadata = {
                "file_path": file_info.get("path", "unknown"),
                "filename": file_info.get("filename", "unknown"),
                "extension": file_info.get("extension", "unknown"),
                "chunk_type": metadata.get("chunk_type", "unknown"),
                "language": metadata.get("language", infer_language_from_path(file_info.get("path", ""))),
                "category": metadata.get("category", infer_category_from_type(metadata.get("chunk_type", ""), file_info.get("path", ""))),
                "chunk_index": i,
                "file_hash": file_hash,
                "added_at": datetime.now().isoformat()
            }
            metadatas.append(enriched_metadata)
            ids.append(chunk_id)

            # Update index
            self._index['extension'][enriched_metadata["extension"]].append(chunk_id)
            self._index['chunk_type'][enriched_metadata["chunk_type"]].append(chunk_id)
            self._index['file_path'][enriched_metadata["file_path"]].append(chunk_id)
            self._index['language'][enriched_metadata["language"]].append(chunk_id)

        # Ajouter à la collection
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        self._save_index()
        logger.info(f"✅ {len(chunks)} chunks ajoutés pour {file_info.get('filename', 'unknown')}")

    def get_all_chunks(self, limit: int = 2000) -> List[CodeChunk]:
        results = self.collection.get(limit=limit, include=["documents", "metadatas", "ids"])
        return [self.create_code_chunk(r) for r in self._format_get_results(results)]
