from pathlib import Path
import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class JavaScriptAnalyzer(Analyzer):
    language = "javascript"

    FUNCTION_RE = re.compile(r"function\s+(\w+)\s*\(")
    CLASS_RE = re.compile(r"class\s+(\w+)")
    IMPORT_RE = re.compile(r"import\s+.*from\s+['\"](.+)['\"]")

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            if match := self.IMPORT_RE.search(line):
                imports.append(match.group(1))

            if match := self.FUNCTION_RE.search(line):
                name = match.group(1)
                symbols.append(name)
                chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line,
                        file_path=str(path),
                        start_line=idx,
                        end_line=idx,
                    )
                )

            if match := self.CLASS_RE.search(line):
                name = match.group(1)
                symbols.append(name)
                chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=name,
                        content=line,
                        file_path=str(path),
                        start_line=idx,
                        end_line=idx,
                    )
                )

        if not chunks:
            chunks.append(
                CodeChunk(
                    language=self.language,
                    name="module",
                    content=content,
                    file_path=str(path),
                    start_line=1,
                    end_line=len(lines),
                )
            )

        return AnalysisResult(
            language=self.language,
            file_path=path,
            chunks=chunks,
            imports=imports,
            symbols=symbols,
            errors=errors,
        )
