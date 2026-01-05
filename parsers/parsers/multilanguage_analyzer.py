from datetime import datetime
from pathlib import Path
from typing import List, Type

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk
from parsers.lang.python_analyzer import PythonAnalyzer
from parsers.lang.javascript_analyzer import JavaScriptAnalyzer
from parsers.lang.vuejs_analyzer import VueJSAnalyzer
from parsers.lang.html_analyzer import HTMLAnalyzer
from parsers.lang.css_analyzer import CSSAnalyzer
from parsers.lang.sql_analyzer import SQLAnalyzer
from parsers.lang.json_analyzer import JSONAnalyzer
from parsers.lang.xml_analyzer import XMLAnalyzer
import logging

logger = logging.getLogger(__name__)

class MultilanguageAnalyzer(Analyzer):
    """
    Analyzer universel qui délègue automatiquement au bon Analyzer
    selon l'extension de fichier.
    """
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

    def analyze_file(self, file_path: str) -> AnalysisResult:
        path=Path(file_path)
        analyzer_cls = self.ANALYZERS.get(path.suffix.lower())
        if analyzer_cls is None:
            # fallback : chunk global
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Mesurer le temps de traitement
                start_time = datetime.now()

                # Créer le résultat de base
                result = self._create_base_result(file_path)
                result.analyzer_name = "None"


                chunk = CodeChunk(
                    language="unknown",
                    name="file",
                    content=content,
                    file_path=path,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                )
                return AnalysisResult(
                    language="unknown",
                    file_path=path,
                    chunks=[chunk],
                    imports=[],
                    symbols=[],
                    errors=[f"No analyzer for extension {path.suffix}"],
                )
            except Exception as e:
                logger.error(f"Error analyzing properties file {file_path}: {e}")
                return self._create_error_result(file_path, str(e))

        analyzer = analyzer_cls()
        return analyzer.analyze_file(path)
