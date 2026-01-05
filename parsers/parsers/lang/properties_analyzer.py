import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class PropertiesAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("properties")
        # Expressions régulières pour les fichiers properties
        self.PROPERTY_RE = re.compile(r'^([\w\.-]+)\s*=\s*(.+)$')
        self.SECTION_RE = re.compile(r'^\[([^\]]+)\]$')
        self.COMMENT_RE = re.compile(r'^[#!]')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        current_section = None

        for idx, line in enumerate(lines, start=1):
            # Ignorer les commentaires
            if self.COMMENT_RE.match(line.strip()):
                continue

            # Sections [section]
            if match := self.SECTION_RE.search(line):
                current_section = match.group(1)
                result.symbols.append(current_section)
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name=current_section,
                        content=line.strip(),
                        file_path=file_path,
                        start_line=idx,
                        end_line=idx,
                    )
                )

            # Propriétés key=value
            elif match := self.PROPERTY_RE.search(line):
                key = match.group(1)
                value = match.group(2)

                # Créer un nom complet avec section si disponible
                full_name = f"{current_section}.{key}" if current_section else key
                result.symbols.append(full_name)

                # Créer un chunk pour les propriétés importantes
                if any(important in key.lower() for important in
                       ['host', 'port', 'database', 'url', 'password', 'username',
                        'config', 'path', 'file', 'dir', 'mode']):
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=key,
                            content=f"{key}={value[:50]}",
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

        return result