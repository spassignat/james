# src/chunk_strategies.py
import logging
from typing import List, Dict, Any

from file.file_info import FileInfo
from parsers.analysis_result import AnalysisResult
from vector.chunk.chunk_strategy import ChunkStrategy

logger = logging.getLogger(__name__)


class JavaScriptChunkStrategy(ChunkStrategy):
    def create_chunks(self, analysis: AnalysisResult, file_info: FileInfo) -> List[Dict[str, Any]]:
        chunks = []

        # Chunk imports (TOUJOURS créer, même si vide)
        imports_chunk = self._create_imports_chunk(analysis, file_info)
        chunks.append(imports_chunk)

        # Chunk fonctions
        for function in analysis.get('functions', []):
            function_chunk = self._create_function_chunk(function, analysis, file_info)
            chunks.append(function_chunk)

        # Chunk classes
        for class_info in analysis.get('classes', []):
            class_chunk = self._create_class_chunk(class_info, analysis, file_info)
            chunks.append(class_chunk)

        # Chunk pour le contenu global (si pas de fonctions/classes)
        if not analysis.get('functions') and not analysis.get('classes'):
            global_chunk = self._create_global_chunk(analysis, file_info)
            chunks.append(global_chunk)

        return chunks

    def _create_global_chunk(self, analysis: Dict, file_info: FileInfo) -> Dict[str, Any]:
        """Crée un chunk pour le contenu global du fichier"""
        content = f"CONTENU GLOBAL JAVASCRIPT: {file_info.filename}\n"
        content += f"FICHIER: {file_info.relative_path}\n"

        # Ajouter des informations sur le type de module
        module_type = analysis.get('analysis', {}).get('module_type', 'unknown')
        content += f"TYPE DE MODULE: {module_type}\n"

        # Ajouter des métadonnées sur le contenu
        exports = analysis.get('exports', [])
        if exports:
            content += f"EXPORTS: {', '.join(exports[:5])}\n"
        else:
            content += "AUCUN EXPORT\n"

        return {
            'content': content,
            'type': 'js_global',
            'metadata': {
                'file_path': file_info.path,
                'module_type': module_type,
                'has_exports': bool(exports),
                'export_count': len(exports)
            }
        }

    def _create_imports_chunk(self, analysis: AnalysisResult, file_info: FileInfo) -> Dict[str, Any]:
        """Crée un chunk pour les imports"""
        content = f"IMPORTS JAVASCRIPT: {file_info.filename}\n"
        content += f"FICHIER: {file_info.relative_path}\n"

        imports = analysis.imports
        if imports:
            content += "IMPORTATIONS:\n"
            for imp in imports[:8]:
                content += f"- {imp}\n"
        else:
            content += "AUCUNE IMPORTATION\n"

        return {
            'content': content,
            'type': 'js_imports',
            'metadata': {
                'file_path': file_info.path,
                'import_count': len(imports),
                'module_type': analysis.get('analysis', {}).get('module_type', 'unknown')
            }
        }

    def _create_class_chunk(self, class_info: Dict, analysis: Dict, file_info: FileInfo) -> Dict[str, Any]:
        """Crée un chunk pour une classe"""
        content = f"CLASSE JAVASCRIPT: {class_info.get('name', 'Anonymous')}\n"
        content += f"FICHIER: {file_info.relative_path}\n"

        # Méthodes de la classe
        methods = class_info.get('methods', [])
        if methods:
            content += f"MÉTHODES: {', '.join(methods[:6])}\n"

        # Détection du type de classe
        if any(method.startswith('render') for method in methods):
            content += "TYPE: Composant React/Vue\n"
        elif any(method.startswith('use') for method in methods):
            content += "TYPE: Hook React\n"

        return {
            'content': content,
            'type': 'js_class',
            'metadata': {
                'class_name': class_info.get('name'),
                'method_count': len(methods),
                'file_path': file_info.path,
                'is_component': any(method.startswith('render') for method in methods)
            }
        }

    def _create_function_chunk(self, function: Dict, analysis: Dict, file_info: FileInfo) -> Dict[str, Any]:
        content = f"FONCTION JS: {function.get('name', 'Anonymous')}\n"
        content += f"FICHIER: {file_info.relative_path}\n"
        content += f"TYPE: {function.get('type', 'function')}\n"

        return {
            'content': content,
            'type': 'js_function',
            'metadata': {
                'function_name': function.get('name'),
                'function_type': function.get('type'),
                'file_path': file_info.path
            }
        }