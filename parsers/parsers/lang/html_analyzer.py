from pathlib import Path
import re
from typing import List

from parsers.parsers.analyzer import Analyzer
from parsers.parsers.analysis_result import AnalysisResult
from model.models.code_chunk import CodeChunk


class HTMLAnalyzer(Analyzer):
    language = "html"

    BLOCK_RE = re.compile(
        r"<(script|style|template)(.*?)>(.*?)</\1>",
        re.DOTALL | re.IGNORECASE,
        )

    SCRIPT_SRC_RE = re.compile(
        r"<script[^>]*src=['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    CSS_LINK_RE = re.compile(
        r"<link[^>]*rel=['\"]stylesheet['\"][^>]*href=['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    TAG_RE = re.compile(r"<([a-zA-Z0-9\-]+)")

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        # Imports JS
        imports.extend(self.SCRIPT_SRC_RE.findall(content))

        # Imports CSS
        imports.extend(self.CSS_LINK_RE.findall(content))

        # Tags HTML (top-level symbols)
        symbols.extend(set(self.TAG_RE.findall(content)))

        # Blocks <script>, <style>, <template>
        for match in self.BLOCK_RE.finditer(content):
            block_type = match.group(1).lower()
            block_content = match.group(3).strip()

            start_line = content[: match.start()].count("\n") + 1
            end_line = start_line + block_content.count("\n")

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=f"html_{block_type}",
                    content=block_content,
                    file_path=path,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        # Fallback : tout le document
        if not chunks:
            chunks.append(
                CodeChunk(
                    language=self.language,
                    name="html_document",
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
