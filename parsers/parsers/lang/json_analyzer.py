from pathlib import Path
import json
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class JSONAnalyzer(Analyzer):
    language = "json"

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(str(e))
            return AnalysisResult(
                language=self.language,
                file_path=path,
                chunks=[],
                imports=[],
                symbols=[],
                errors=errors,
            )

        if isinstance(data, dict):
            symbols.extend(data.keys())

        chunks.append(
            CodeChunk(
                language=self.language,
                name="json_document",
                content=content,
                file_path=str(path),
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
