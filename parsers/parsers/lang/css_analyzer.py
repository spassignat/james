import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class CSSAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("css")
        # Expressions régulières pour CSS
        self.SELECTOR_RE = re.compile(r'([^{]+)\s*{')
        self.CLASS_RE = re.compile(r'\.([\w-]+)')
        self.ID_RE = re.compile(r'#([\w-]+)')
        self.IMPORT_RE = re.compile(r'@import\s+["\']([^"\']+)["\']')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            # @import
            if match := self.IMPORT_RE.search(line):
                result.imports.append(match.group(1))

            # Sélecteurs CSS
            if match := self.SELECTOR_RE.search(line):
                selector = match.group(1).strip()
                result.symbols.append(selector)

                # Classes et IDs dans le sélecteur
                for class_match in self.CLASS_RE.findall(selector):
                    result.symbols.append(f".{class_match}")

                for id_match in self.ID_RE.findall(selector):
                    result.symbols.append(f"#{id_match}")

                # Créer un chunk pour le sélecteur
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=selector.split()[0] if selector else "selector",
                        content=selector,
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

        return result