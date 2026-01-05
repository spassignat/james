from typing import Dict, Any, List

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult
from vector.chunk.chunk_strategy import ChunkStrategy


class CSSChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        chunks = []
        # Implémentation basique pour CSS
        content = f"CSS CONTENT: {file_info.filename}\n"
        content += f"Règles détectées: {analysis.rule_count or 0}"

        chunks.append({
            'content': content,
            'type': 'css_content',
            'metadata': {'file_path': file_info.path}
        })
        return chunks