import re
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class HTMLAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("html")
        # Expressions régulières pour HTML
        self.TAG_RE = re.compile(r'<(\w+)[^>]*>')
        self.CLASS_RE = re.compile(r'class\s*=\s*["\']([^"\']+)["\']')
        self.ID_RE = re.compile(r'id\s*=\s*["\']([^"\']+)["\']')
        self.SCRIPT_RE = re.compile(r'<script[^>]*src\s*=\s*["\']([^"\']+)["\']')
        self.LINK_RE = re.compile(r'<link[^>]*href\s*=\s*["\']([^"\']+)["\']')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            # Scripts et liens externes
            if match := self.SCRIPT_RE.search(line):
                result.imports.append(match.group(1))

            if match := self.LINK_RE.search(line):
                result.imports.append(match.group(1))

            # Balises HTML
            for match in self.TAG_RE.finditer(line):
                tag = match.group(1)
                result.symbols.append(tag)

                # Créer un chunk pour les balises importantes
                if tag in ['div', 'span', 'section', 'article', 'header', 'footer',
                           'nav', 'main', 'aside', 'form', 'input', 'button']:
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=tag,
                            content=line.strip(),
                            file_path=file_path,
                            start_line=idx,
                            end_line=idx,
                        )
                    )

            # Classes et IDs
            for class_match in self.CLASS_RE.findall(line):
                classes = class_match.split()
                for cls in classes:
                    result.symbols.append(f".{cls}")

            for id_match in self.ID_RE.findall(line):
                result.symbols.append(f"#{id_match}")

        return result