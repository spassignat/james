import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Type

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk
from parsers.lang.css_analyzer import CSSAnalyzer
from parsers.lang.html_analyzer import HTMLAnalyzer
from parsers.lang.javascript_analyzer import JavaScriptAnalyzer
from parsers.lang.json_analyzer import JSONAnalyzer
from parsers.lang.python_analyzer import PythonAnalyzer
from parsers.lang.sql_analyzer import SQLAnalyzer
from parsers.lang.vuejs_analyzer import VueJSAnalyzer
from parsers.lang.xml_analyzer import XMLAnalyzer

logger = logging.getLogger(__name__)


class MultilanguageAnalyzer(Analyzer):
    """
    Analyzer universel qui délègue automatiquement au bon Analyzer
    selon l'extension de fichier.
    """

    def __init__(self):
        super().__init__("multi")

    ANALYZERS: dict[str, Type[Analyzer]] = {
        ".py": PythonAnalyzer,
        ".js": JavaScriptAnalyzer,
        ".vue": VueJSAnalyzer,
        ".html": HTMLAnalyzer,
        ".htm": HTMLAnalyzer,
        ".css": CSSAnalyzer,
        ".sql": SQLAnalyzer,
        ".json": JSONAnalyzer,
        ".xml": XMLAnalyzer,
    }

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        pass

    def analyze_file(self, file_path: str) -> AnalysisResult:
        path = Path(file_path)
        analyzer_cls = self.ANALYZERS.get(path.suffix.lower())
        if analyzer_cls is None:
            # fallback : chunk global
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Créer le résultat de base
                result = AnalysisResult(language=self.language)

                chunk = CodeChunk(
                    language="unknown",
                    name="file",
                    content=content,
                    file_path=path,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                )
                result.chunks = [chunk]
                return result
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
                return self._create_error_result(Path(file_path), str(e))

        try:
            logger.info(f"Analyzing file {file_path} using : {analyzer_cls}")
            analyzer = analyzer_cls()
            return analyzer.analyze_file(str(path))
        except:
            print(traceback.format_exc())
        return AnalysisResult(self.language)
