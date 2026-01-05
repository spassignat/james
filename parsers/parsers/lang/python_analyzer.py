import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class PythonAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("python")
        # Expressions régulières pour Python
        self.FUNCTION_RE = re.compile(r'def\s+(\w+)\s*\(')
        self.CLASS_RE = re.compile(r'class\s+(\w+)')
        self.IMPORT_RE = re.compile(r'import\s+([\w\.]+)')
        self.FROM_IMPORT_RE = re.compile(r'from\s+([\w\.]+)\s+import')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            # Imports
            if match := self.IMPORT_RE.search(line):
                result.imports.append(match.group(1))

            if match := self.FROM_IMPORT_RE.search(line):
                result.imports.append(match.group(1))

            # Fonctions
            if match := self.FUNCTION_RE.search(line):
                name = match.group(1)
                result.symbols.append(name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line.strip(),
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Classes
            if match := self.CLASS_RE.search(line):
                name = match.group(1)
                result.symbols.append(name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line.strip(),
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

        return result