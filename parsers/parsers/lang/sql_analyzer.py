import re

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class SQLAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("sql")
        # Expressions régulières pour SQL
        self.CREATE_TABLE_RE = re.compile(r'CREATE\s+TABLE\s+(\w+)', re.IGNORECASE)
        self.CREATE_VIEW_RE = re.compile(r'CREATE\s+VIEW\s+(\w+)', re.IGNORECASE)
        self.SELECT_RE = re.compile(r'SELECT\s+.*?\s+FROM\s+(\w+)', re.IGNORECASE)
        self.INSERT_RE = re.compile(r'INSERT\s+INTO\s+(\w+)', re.IGNORECASE)
        self.UPDATE_RE = re.compile(r'UPDATE\s+(\w+)', re.IGNORECASE)
        self.DELETE_RE = re.compile(r'DELETE\s+FROM\s+(\w+)', re.IGNORECASE)

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            # CREATE TABLE
            if match := self.CREATE_TABLE_RE.search(line):
                table_name = match.group(1)
                result.symbols.append(table_name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=table_name,
                        content=line.strip()[:100],
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # CREATE VIEW
            if match := self.CREATE_VIEW_RE.search(line):
                view_name = match.group(1)
                result.symbols.append(view_name)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=view_name,
                        content=line.strip()[:100],
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # SELECT (tables sources)
            if match := self.SELECT_RE.search(line):
                table_name = match.group(1)
                result.symbols.append(table_name)

            # INSERT INTO
            if match := self.INSERT_RE.search(line):
                table_name = match.group(1)
                result.symbols.append(table_name)

            # UPDATE
            if match := self.UPDATE_RE.search(line):
                table_name = match.group(1)
                result.symbols.append(table_name)

            # DELETE FROM
            if match := self.DELETE_RE.search(line):
                table_name = match.group(1)
                result.symbols.append(table_name)

        return result