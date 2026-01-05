from typing import Dict, Any, List

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult


class ChunkStrategy:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        """Crée des chunks à partir de l'analyse d'un fichier"""
        raise NotImplementedError
