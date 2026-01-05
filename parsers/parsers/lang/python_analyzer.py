from pathlib import Path
import ast
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class PythonAnalyzer(Analyzer):
    language = "python"

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            errors.append(str(e))
            return AnalysisResult(
                language=self.language,
                file_path=path,
                chunks=[],
                imports=[],
                symbols=[],
                errors=errors,
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

            elif isinstance(node, ast.FunctionDef):
                symbols.append(node.name)
                chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=node.name,
                        content=ast.get_source_segment(content, node) or "",
                        file_path=str(path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                    )
                )

            elif isinstance(node, ast.ClassDef):
                symbols.append(node.name)
                chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=node.name,
                        content=ast.get_source_segment(content, node) or "",
                        file_path=str(path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
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
