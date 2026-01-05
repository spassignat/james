# src/chunk_strategies.py
import logging
from typing import List, Dict, Any

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult
from vector.chunk.chunk_strategy import ChunkStrategy

logger = logging.getLogger(__name__)


class GenericChunkStrategy(ChunkStrategy):
    """Stratégie générique pour les fichiers sans analyseur spécifique"""

    def create_chunks(self, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        content = f"FICHIER: {file_info.relative_path}\n"
        content += f"TYPE: {analysis.file_type or 'unknown'}\n"
        content += f"TAILLE: {file_info.size} bytes\n"

        if analysis.content:
            # Découpage intelligent du contenu
            text_content = analysis. content
            if len(text_content) > self.chunk_size:
                text_content = text_content[:self.chunk_size] + "..."
            content += f"CONTENU:\n{text_content}\n"

        return [{
            'content': content,
            'type': 'generic',
            'metadata': {
                'file_path': file_info.path,
                'file_type': analysis.get('file_type'),
                'line_count': analysis.get('line_count', 0)
            }
        }]
