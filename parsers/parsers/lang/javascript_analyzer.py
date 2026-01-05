import re

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class JavaScriptAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("javascript")

    FUNCTION_RE = re.compile(r"function\s+(\w+)\s*\(")
    CLASS_RE = re.compile(r"class\s+(\w+)")
    IMPORT_RE = re.compile(r"import\s+.*from\s+['\"](.+)['\"]")

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)

        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            if match := self.IMPORT_RE.search(line):
                result.imports.append(match.group(1))

            if match := self.FUNCTION_RE.search(line):
                name = match.group(1)
                result.symbols.append(name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line,
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            if match := self.CLASS_RE.search(line):
                name = match.group(1)
                result.symbols.append(name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line,
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

        return result

