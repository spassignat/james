# src/chunk_strategies.py
import logging
from typing import List, Dict, Any

from vector.chunk.chunk_strategy import ChunkStrategy

logger = logging.getLogger(__name__)


class JavaChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: Dict[str, Any], file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []

        # Chunk 1: Vue d'ensemble de la classe
        if analysis.get('classes'):
            for class_info in analysis.get('classes', []):
                chunk_content = self._create_class_chunk(analysis, class_info, file_info)
                chunks.append(chunk_content)

        # Chunks par méthode
        for method in analysis.get('methods', []):
            method_chunk = self._create_method_chunk(method, analysis, file_info)
            chunks.append(method_chunk)

        # Chunk pour les champs importants
        if analysis.get('fields'):
            fields_chunk = self._create_fields_chunk(analysis, file_info)
            chunks.append(fields_chunk)

        return chunks

    def _create_class_chunk(self, analysis: Dict, class_info: Dict, file_info: Dict) -> Dict[str, Any]:
        content = f"CLASSE JAVA: {class_info.get('name', 'Unknown')}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"
        content += f"PAQUET: {analysis.get('package', '')}\n"

        if analysis.get('imports'):
            content += f"IMPORTS: {', '.join(analysis.get('imports', [])[:10])}\n"

        if class_info.get('modifiers'):
            content += f"MODIFICATEURS: {class_info.get('modifiers')}\n"

        return {
            'content': content,
            'type': 'java_class',
            'metadata': {
                'class_name': class_info.get('name'),
                'package': analysis.get('package'),
                'file_path': file_info['path']
            }
        }

    def _create_method_chunk(self, method: Dict, analysis: Dict, file_info: Dict) -> Dict[str, Any]:
        content = f"MÉTHODE JAVA: {method.get('name', 'Unknown')}\n"
        content += f"CLASSE: {analysis.get('classes', [{}])[0].get('name', 'Unknown')}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"
        content += f"VISIBILITÉ: {method.get('visibility', '')}\n"

        if method.get('annotations'):
            content += f"ANNOTATIONS: {', '.join(method.get('annotations', []))}\n"

        if method.get('is_constructor', False):
            content += "TYPE: Constructeur\n"

        return {
            'content': content,
            'type': 'java_method',
            'metadata': {
                'method_name': method.get('name'),
                'class_name': analysis.get('classes', [{}])[0].get('name'),
                'visibility': method.get('visibility')
            }
        }

    def _create_fields_chunk(self, analysis: Dict, file_info: Dict) -> Dict[str, Any]:
        content = f"CHAMPS JAVA - CLASSE: {analysis.get('classes', [{}])[0].get('name', 'Unknown')}\n"
        content += f"FICHIER: {file_info['relative_path']}\n"

        for field in analysis.get('fields', [])[:5]:  # Limiter aux 5 premiers
            content += f"- {field.get('name')} : {field.get('type')} ({field.get('visibility')})\n"

        return {
            'content': content,
            'type': 'java_fields',
            'metadata': {
                'class_name': analysis.get('classes', [{}])[0].get('name'),
                'field_count': len(analysis.get('fields', []))
            }
        }
