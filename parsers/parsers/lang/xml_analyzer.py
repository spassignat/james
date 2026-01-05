from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class XMLAnalyzer(Analyzer):
    language = "xml"

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            errors.append(str(e))
            return AnalysisResult(
                language=self.language,
                file_path=path,
                chunks=[],
                imports=[],
                symbols=[],
                errors=errors,
            )

        symbols.append(root.tag)

        start_line = 1
        end_line = content.count("\n") + 1

        chunks.append(
            CodeChunk(
                language=self.language,
                name=root.tag,
                content=content,
                file_path=str(path),
                start_line=start_line,
                end_line=end_line,
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
