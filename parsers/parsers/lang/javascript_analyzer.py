import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class JavaScriptAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("javascript")

        # Expressions régulières pour JavaScript/TypeScript/Vue
        self.FUNCTION_RE = re.compile(r'\b(function\s+(\w+)\s*\(|(\w+)\s*=\s*function\s*\()')
        self.ARROW_FUNCTION_RE = re.compile(r'\b(const|let|var)?\s*(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>')
        self.CLASS_RE = re.compile(r'\b(class|interface)\s+(\w+)')
        self.IMPORT_RE = re.compile(r'(?:import|require)\s*(?:.*?from\s*)?[\'"]([^\'"]+)[\'"]')

        # Pour les modules Vue.js
        self.VUE_COMPONENT_RE = re.compile(r'export\s+default\s*{')
        self.VUE_LIFECYCLE_RE = re.compile(r'\b(beforeCreate|created|beforeMount|mounted|beforeUpdate|updated'
                                           r'|beforeDestroy|destroyed|activated|deactivated|errorCaptured)\s*\(\s*\)')
        self.VUE_METHODS_RE = re.compile(r'methods\s*:\s*{')
        self.VUE_METHOD_DEF_RE = re.compile(r'\b(\w+)\s*\(\s*\)\s*{')

        # Pour les objets et propriétés
        self.PROPERTY_RE = re.compile(r'\b(\w+)\s*:\s*[^{]')
        self.OBJECT_METHOD_RE = re.compile(r'\b(\w+)\s*\(\s*[^)]*\)\s*{')

        # Variables importantes
        self.VARIABLE_RE = re.compile(r'\b(const|let|var)\s+(\w+)\s*=')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        in_vue_component = False
        in_vue_methods = False
        vue_component_name = self._guess_component_name(file_path)

        for idx, line in enumerate(lines, start=1):
            line_stripped = line.strip()

            # Détecter si on est dans un composant Vue.js
            if self.VUE_COMPONENT_RE.search(line_stripped):
                in_vue_component = True
                if vue_component_name:
                    result.symbols.append(vue_component_name)
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=vue_component_name,
                            content="Vue.js Component",
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Imports
            for match in self.IMPORT_RE.findall(line_stripped):
                result.imports.append(match)

            # Détecter le début/fin de la section methods
            if in_vue_component and self.VUE_METHODS_RE.search(line_stripped):
                in_vue_methods = True
            elif in_vue_methods and '}' in line_stripped and '{' not in line_stripped:
                in_vue_methods = False

            # Méthodes du cycle de vie Vue.js
            if in_vue_component and (match := self.VUE_LIFECYCLE_RE.search(line_stripped)):
                method_name = match.group(1)
                result.symbols.append(method_name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=method_name,
                        content=line_stripped[:150],
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Méthodes dans la section methods
            if in_vue_methods and (match := self.VUE_METHOD_DEF_RE.search(line_stripped)):
                method_name = match.group(1)
                result.symbols.append(method_name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=method_name,
                        content=line_stripped[:150],
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Fonctions traditionnelles
            if match := self.FUNCTION_RE.search(line_stripped):
                name = match.group(2) or match.group(3)
                if name:
                    result.symbols.append(name)
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=name,
                            content=line_stripped[:150],
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Fonctions fléchées
            if match := self.ARROW_FUNCTION_RE.search(line_stripped):
                name = match.group(2)
                if name:
                    result.symbols.append(name)
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=name,
                            content=line_stripped[:150],
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Classes et interfaces
            if match := self.CLASS_RE.search(line_stripped):
                name = match.group(2)
                result.symbols.append(name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line_stripped[:150],
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Méthodes d'objet (hors Vue)
            if not in_vue_component and (match := self.OBJECT_METHOD_RE.search(line_stripped)):
                method_name = match.group(1)
                # Éviter les mots-clés
                if method_name not in ['if', 'for', 'while', 'switch', 'try', 'catch']:
                    result.symbols.append(method_name)
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=method_name,
                            content=line_stripped[:150],
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Propriétés d'objet
            if match := self.PROPERTY_RE.search(line_stripped):
                prop_name = match.group(1)
                if prop_name not in ['const', 'let', 'var', 'if', 'for', 'while']:
                    result.symbols.append(prop_name)

            # Variables importantes
            if match := self.VARIABLE_RE.search(line_stripped):
                var_name = match.group(2)
                # Capturer les variables importantes
                if var_name in ['app', 'port', 'server', 'router', 'db', 'config',
                                'service', 'api', 'component', 'store', 'vue']:
                    result.symbols.append(var_name)
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=var_name,
                            content=line_stripped[:150],
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

        return result

    def _guess_component_name(self, file_path: str) -> str:
        """Devine le nom du composant à partir du chemin du fichier"""
        import os
        filename = os.path.basename(file_path)

        # Supprimer l'extension
        name_without_ext = os.path.splitext(filename)[0]

        # Convertir en PascalCase si c'est en kebab-case
        if '-' in name_without_ext:
            parts = name_without_ext.split('-')
            return ''.join(part.capitalize() for part in parts)

        return name_without_ext