import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class VueJSAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("vue")
        # Expressions régulières pour Vue.js
        self.TEMPLATE_RE = re.compile(r'<template[^>]*>([\s\S]*?)</template>')
        self.SCRIPT_RE = re.compile(r'<script[^>]*>([\s\S]*?)</script>')
        self.STYLE_RE = re.compile(r'<style[^>]*>([\s\S]*?)</style>')

        # Pour le template (HTML)
        self.TAG_RE = re.compile(r'<([\w-]+)[^>]*>')
        self.VUE_DIRECTIVE_RE = re.compile(r'v-([\w-]+)')
        self.VUE_BIND_RE = re.compile(r':([\w-]+)')
        self.VUE_EVENT_RE = re.compile(r'@([\w-]+)')

        # Pour le script (JavaScript)
        self.COMPONENT_NAME_RE = re.compile(r'name\s*:\s*["\']([^"\']+)["\']')
        self.METHOD_RE = re.compile(r'methods\s*:\s*{([^}]+)}')
        self.PROP_RE = re.compile(r'props\s*:\s*{([^}]+)}')
        self.DATA_RE = re.compile(r'data\s*\(\s*\)\s*{\s*return\s*{([^}]+)}')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)

        # Extraire les sections
        template_match = self.TEMPLATE_RE.search(content)
        script_match = self.SCRIPT_RE.search(content)
        style_match = self.STYLE_RE.search(content)

        # Analyser le template (HTML/Vue)
        if template_match:
            template_content = template_match.group(1)
            self._analyze_template(template_content, file_path, result)

        # Analyser le script (JavaScript)
        if script_match:
            script_content = script_match.group(1)
            self._analyze_script(script_content, file_path, result)

        # Analyser le style (CSS) - optionnel
        if style_match:
            # On pourrait analyser le CSS ici, mais pour simplifier, on l'ignore
            pass

        return result

    def _analyze_template(self, template: str, file_path: str, result: AnalysisResult):
        """Analyse la section template"""
        lines = template.splitlines()

        for idx, line in enumerate(lines, start=1):
            # Balises Vue/HTML
            for match in self.TAG_RE.finditer(line):
                tag = match.group(1)
                result.symbols.append(tag)

                # Créer un chunk pour les composants Vue
                if tag[0].isupper() or tag.startswith('v-') or tag in ['template', 'slot', 'component']:
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=tag,
                            content=line.strip()[:100],
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Directives Vue
            for match in self.VUE_DIRECTIVE_RE.finditer(line):
                directive = f"v-{match.group(1)}"
                result.symbols.append(directive)

            # Bindings Vue (:)
            for match in self.VUE_BIND_RE.finditer(line):
                binding = f":{match.group(1)}"
                result.symbols.append(binding)

            # Événements Vue (@)
            for match in self.VUE_EVENT_RE.finditer(line):
                event = f"@{match.group(1)}"
                result.symbols.append(event)

    def _analyze_script(self, script: str, file_path: str, result: AnalysisResult):
        """Analyse la section script"""
        lines = script.splitlines()

        for idx, line in enumerate(lines, start=1):
            # Nom du composant
            if match := self.COMPONENT_NAME_RE.search(line):
                component_name = match.group(1)
                result.symbols.append(component_name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=component_name,
                        content=f"Component: {component_name}",
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Méthodes
            if 'methods:' in line or 'methods :' in line:
                # Rechercher les méthodes dans les lignes suivantes
                for j in range(idx, min(idx + 20, len(lines))):
                    method_line = lines[j]
                    if ':' in method_line and '{' not in method_line:
                        method_name = method_line.split(':')[0].strip()
                        if method_name and method_name not in ['', 'methods']:
                            result.symbols.append(method_name)
                            result.chunks.append(
                                CodeChunk(
                                    language=self.language,
                                    name=method_name,
                                    content=f"method: {method_name}",
                                    file_path=file_path,
                                    start_line=j,
                                    end_line=j,
                                )
                            )

            # Props
            if 'props:' in line or 'props :' in line:
                # Rechercher les props dans les lignes suivantes
                for j in range(idx, min(idx + 20, len(lines))):
                    prop_line = lines[j]
                    if ':' in prop_line and '{' not in prop_line:
                        prop_name = prop_line.split(':')[0].strip()
                        if prop_name and prop_name not in ['', 'props']:
                            result.symbols.append(prop_name)