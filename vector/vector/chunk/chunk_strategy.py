from typing import Dict, Any, List


class ChunkStrategy:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, analysis: Dict[str, Any], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crée des chunks à partir de l'analyse d'un fichier"""
        raise NotImplementedError
