# multi_language_analyzer.py
import logging
import os
from typing import Dict, Any

from james_parsers.analyzer import Analyzer
from james_parsers.css_analyzer import CSSAnalyzer
from james_parsers.generic_analyzer import GenericAnalyzer
from james_parsers.html_analyzer import HTMLAnalyzer
from james_parsers.java_analyzer import JavaAnalyzer
from james_parsers.javascript_analyzer import JavaScriptAnalyzer
from james_parsers.json_analyzer import JSONAnalyzer
from james_parsers.properties_analyzer import PropertiesAnalyzer
from james_parsers.python_analyzer import PythonAnalyzer
from james_parsers.sql_analyzer import SQLAnalyzer
from james_parsers.vuejs_analyzer import VueJSAnalyzer
from james_parsers.xml_analyzer import XMLAnalyzer

logger = logging.getLogger(__name__)


class MultiLanguageAnalyzer(Analyzer):
    def __init__(self, config):
        self.config = config
        self.analyzers = {
            '.java': JavaAnalyzer(),
            '.vue': VueJSAnalyzer(),
            '.js': JavaScriptAnalyzer(),
            '.sql': SQLAnalyzer(),
            '.json': JSONAnalyzer(),
            '.properties': PropertiesAnalyzer(),
            '.xml': XMLAnalyzer(),
            '.html': HTMLAnalyzer(),
            '.css': CSSAnalyzer(),
            '.py': PythonAnalyzer()
        }

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyse un fichier avec l'analyseur approprié"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in self.analyzers:
            return self.analyzers[ext].analyze(file_path)
        else:
            return GenericAnalyzer().analyze(file_path)
