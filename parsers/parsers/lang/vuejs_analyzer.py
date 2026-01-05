from pathlib import Path
import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class VueJSAnalyzer(Analyzer):
    language = "vue"

    BLOCK_RE = re.compile(r"<(template|script|style)(.*?)>(.*?)</\1>", re.DOTALL)
    IMPORT_RE = re.compile(r"import\s+.*from\s+['\"](.+)['\"]")

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        for match in self.BLOCK_RE.finditer(content):
            block_type = match.group(1)
            block_content = match.group(3).strip()

            start_line = content[: match.start()].count("\n") + 1
            end_line = start_line + block_content.count("\n")

            name = f"vue_{block_type}"

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=name,
                    content=block_content,
                    file_path=str(path),
                    start_line=start_line,
                    end_line=end_line,
                )
            )

            if block_type == "script":
                for imp in self.IMPORT_RE.findall(block_content):
                    imports.append(imp)

        if not chunks:
            errors.append("No Vue blocks detected")

        return AnalysisResult(
            language=self.language,
            file_path=path,
            chunks=chunks,
            imports=imports,
            symbols=symbols,
            errors=errors,
        )
