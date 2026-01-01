# Stratégies simplifiées pour la délégation
from typing import Dict, Any, List

from vector.chunk.chunk_strategy import ChunkStrategy


class HTMLChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: Dict[str, Any], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        # Implémentation basique pour HTML
        content = f"HTML CONTENT: {file_info['filename']}\n"
        content += f"Éléments détectés: {len(analysis.get('elements', []))}"

        chunks.append({
            'content': content,
            'type': 'html_content',
            'metadata': {'file_path': file_info['path']}
        })
        return chunks
