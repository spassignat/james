from pathlib import Path
import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class SQLAnalyzer(Analyzer):
    language = "sql"

    STATEMENT_RE = re.compile(
        r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b",
        re.IGNORECASE,
    )
    TABLE_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_]+)", re.IGNORECASE)

    def analyze(self, path: Path, content: str) -> AnalysisResult:
        chunks: List[CodeChunk] = []
        imports: List[str] = []
        symbols: List[str] = []
        errors: List[str] = []

        statements = content.split(";")

        for idx, stmt in enumerate(statements, start=1):
            stmt = stmt.strip()
            if not stmt:
                continue

            kind_match = self.STATEMENT_RE.search(stmt)
            name = kind_match.group(1).upper() if kind_match else f"statement_{idx}"

            for table in self.TABLE_RE.findall(stmt):
                symbols.append(table)

            start_line = content[: content.find(stmt)].count("\n") + 1
            end_line = start_line + stmt.count("\n")

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=name,
                    content=stmt,
                    file_path=str(path),
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        if not chunks:
            errors.append("No SQL statements detected")

        return AnalysisResult(
            language=self.language,
            file_path=path,
            chunks=chunks,
            imports=imports,
            symbols=list(set(symbols)),
            errors=errors,
        )
