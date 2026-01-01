from typing import Dict, Any, List

from vector.chunk.chunk_strategy import ChunkStrategy


class CSSChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: Dict[str, Any], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        # Implémentation basique pour CSS
        content = f"CSS CONTENT: {file_info['filename']}\n"
        content += f"Règles détectées: {analysis.get('rule_count', 0)}"

        chunks.append({
            'content': content,
            'type': 'css_content',
            'metadata': {'file_path': file_info['path']}
        })
        return chunks