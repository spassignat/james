from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, runtime_checkable, Protocol

import time


class ChunkType(str, Enum):
    """Types de chunks standardisés"""
    TEXT = "text"
    CODE = "code"
    NORMALIZED_DICT = "normalized_dict"
    DICT_JSON = "dict_json"
    DICT_FALLBACK = "dict_fallback"
    LIST_JSON = "list_json"
    LIST_FALLBACK = "list_fallback"
    OTHER = "other"
    RAW_FALLBACK = "raw_fallback"
    ERROR = "error"


class FileStatus(str, Enum):
    """Statuts possibles pour le traitement d'un fichier"""
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"
    SKIPPED = "skipped"


@dataclass
class ChunkMetadata:
    """Métadonnées standardisées pour un chunk"""
    file_path: str
    relative_path: str
    filename: str
    extension: str
    chunk_index: int
    chunk_id: str
    chunk_type: ChunkType
    chunk_size: int
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: Optional[str] = None
    category: Optional[str] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    element_type: Optional[str] = None
    modifier: Optional[str] = None
    visibility: Optional[str] = None
    parameters: Optional[str] = None
    modifiers: Optional[str] = None
    attributes: Optional[str] = None
    dependencies: Optional[str] = None
    processed_at: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass
class NormalizedChunk:
    """Structure normalisée pour un chunk"""
    id: str
    content: str
    type: ChunkType
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


@dataclass
class FileProcessingResult:
    """Résultat du traitement d'un fichier"""
    success: bool
    chunks: List[NormalizedChunk]
    error: Optional[str] = None
    fallback_used: bool = False
    analysis_status: Optional[str] = None


@dataclass
class VectorizationStats:
    """Statistiques de vectorisation"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    fallback_files: int = 0
    total_chunks: int = 0
    chunks_by_type: Dict[ChunkType, int] = field(default_factory=lambda: defaultdict(int))
    files_by_extension: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    files_by_status: Dict[FileStatus, int] = field(default_factory=lambda: defaultdict(int))
    total_vector_size: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    scan_error: Optional[str] = None
    persisted: bool = False
    persist_error: Optional[str] = None


@runtime_checkable
class ChunkProtocol(Protocol):
    """Protocole pour les chunks bruts venant de l'analyzer"""
    content: str
