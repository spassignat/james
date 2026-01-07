import re

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class XMLAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("xml")
        # Expressions régulières pour XML
        self.ELEMENT_RE = re.compile(r'<(\w+)[^>]*>')
        self.CLOSING_ELEMENT_RE = re.compile(r'</(\w+)>')
        self.ATTRIBUTE_RE = re.compile(r'(\w+)\s*=\s*["\'][^"\']*["\']')
        self.ROOT_ELEMENT_RE = re.compile(r'<\?xml[^?>]*\?>\s*<(\w+)')

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        lines = content.splitlines()

        # Trouver l'élément racine
        if match := self.ROOT_ELEMENT_RE.search(content):
            root_element = match.group(1)
            result.symbols.append(root_element)
            result.chunks.append(
                CodeChunk(
                    language=self.language,
                    name=root_element,
                    content=f"Root element: {root_element}",
                    file_path=file_path,
                    start_line=1,
                    end_line=1,
                )
            )

        for idx, line in enumerate(lines, start=1):
            # Éléments XML
            for match in self.ELEMENT_RE.finditer(line):
                element = match.group(1)

                # Ignorer les déclarations XML
                if element not in ['?xml', '!DOCTYPE', '!ENTITY']:
                    result.symbols.append(element)

                    # Créer un chunk pour les éléments importants
                    if any(important in element.lower() for important in
                           ['bean', 'component', 'service', 'controller', 'repository',
                            'config', 'property', 'setting', 'element', 'node']):
                        result.chunks.append(
                            CodeChunk(
                                language=self.language,
                                name=element,
                                content=line.strip()[:100],
                                file_path=file_path,
                                start_line=idx,
                                end_line=idx,
                            )
                        )

            # Attributs
            for match in self.ATTRIBUTE_RE.finditer(line):
                attribute = match.group(1)
                result.symbols.append(attribute)

        return result