from pathlib import Path
import re
from typing import List

from parsers.parsers.analyzer import Analyzer
from parsers.parsers.analysis_result import AnalysisResult
from model.models.code_chunk import CodeChunk


class CSSAnalyzer(Analyzer):
    language = "css"

    IMPORT_RE = re.compile(r"@import\s+['\"]([^'\"]+)['\"]")
    RULE_RE = re.compile(r"([^{]+)\{([^}]+)\}", re.DOTALL)

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        imports.extend(self.IMPORT_RE.findall(content))

        for match in self.RULE_RE.finditer(content):
            selector = match.group(1).strip()
            body = match.group(2).strip()

            start_line = content[: match.start()].count("\n") + 1
            end_line = start_line + body.count("\n")

            symbols.append(selector)

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=selector,
                    content=f"{selector} {{\n{body}\n}}",
                    file_path=path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        # Fallback
        if not chunks:
            chunks.append(
                CodeChunk(
                    language=self.language,
                    name="css_stylesheet",
                    content=content,
                    file_path=path,
                    start_line=1,
                    end_line=content.count("\n") + 1,
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
