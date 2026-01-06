import re
from pathlib import Path
from typing import List

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk

SCRIPT_RE = re.compile(r"<script[^>]*>(?P<script>.*?)</script>", re.S)

EXPORT_DEFAULT_RE = re.compile(r"export\s+default\s*\{", re.S)

SECTION_RE = re.compile(
    r"""
    (?P<section>data|methods|computed|props|mixins)
    \s*:\s*
    (?P<body>\{)
    """,
    re.VERBOSE,
)

PROPERTY_RE = re.compile(r"(?P<name>\w+)\s*[:\(]")


class VueJSAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("vuejs")

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        script_match = SCRIPT_RE.search(content)
        path = Path(file_path)
        if not script_match:
            return self._fallback(path, content)

        script = script_match.group("script")
        script_lines = script.splitlines()

        chunks: List[CodeChunk] = []

        component_name = path.stem

        # ---------- Component ----------
        export_match = EXPORT_DEFAULT_RE.search(script)
        if export_match:
            start = self._line_of(script, export_match.start())
            end = self._find_block_end(script_lines, start)

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=f"component {component_name}",
                    content="\n".join(script_lines[start - 1: end]),
                    file_path=path,
                    start_line=start,
                    end_line=end,
                )
            )

        # ---------- Sections ----------
        for match in SECTION_RE.finditer(script):
            section = match.group("section")
            start = self._line_of(script, match.start())
            end = self._find_block_end(script_lines, start)

            self._extract_section(
                path,
                component_name,
                section,
                script_lines,
                start,
                end,
                chunks,
            )

        result = AnalysisResult(language=self.language)
        result.file_path = path
        result.chunks = chunks
        result.symbols = [c.name for c in chunks]
        return result

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _extract_section(
            self,
            path: Path,
            component: str,
            section: str,
            lines: List[str],
            start: int,
            end: int,
            chunks: List[CodeChunk],
    ) -> None:
        body = lines[start: end - 1]
        offset = start + 1

        for i, line in enumerate(body):
            prop_match = PROPERTY_RE.search(line)
            if not prop_match:
                continue

            name = prop_match.group("name")
            line_no = offset + i

            chunk_name = f"{component}.{section}.{name}"
            if section in ("methods", "computed"):
                chunk_name += "()"

            # method / computed → bloc complet
            if section in ("methods", "computed"):
                block_end = self._find_block_end(lines, line_no)
                chunk_content = "\n".join(lines[line_no - 1: block_end])
                chunk_end = block_end
            else:
                # props / data
                chunk_content = line.strip()
                chunk_end = line_no

            chunks.append(
                CodeChunk(
                    language=self.language,
                    name=chunk_name,
                    content=chunk_content,
                    file_path=path,
                    start_line=line_no,
                    end_line=chunk_end,
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_block_end(lines: List[str], start_line: int) -> int:
        brace_count = 0
        for i in range(start_line - 1, len(lines)):
            brace_count += lines[i].count("{")
            brace_count -= lines[i].count("}")
            if brace_count == 0 and i > start_line - 1:
                return i + 1
        return len(lines)

    @staticmethod
    def _line_of(content: str, index: int) -> int:
        return content[:index].count("\n") + 1

    def _fallback(self, path: Path, content: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)
        result.file_path = path
        result.chunks = [
            CodeChunk(language=self.language, name=path.name, content=content, file_path=path, start_line=1,
                      end_line=content.count("\n") + 1, )]
        result.imports = []
        result.symbols = [path.name]
        return result
