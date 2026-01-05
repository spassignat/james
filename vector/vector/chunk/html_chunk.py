# Stratégies simplifiées pour la délégation
from typing import Dict, Any, List

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult
from vector.chunk.chunk_strategy import ChunkStrategy


class HTMLChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        chunks = []
        # Implémentation basique pour HTML
        content = f"HTML CONTENT: {file_info.filename}\n"
        content += f"Éléments détectés: {len(analysis.get('elements', []))}"

        chunks.append({
            'content': content,
            'type': 'html_content',
            'metadata': {'file_path': file_info.path}
        })
        return chunks
