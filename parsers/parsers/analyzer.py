# parsers/analyzer.py
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from file.file_name_parser import FilenameParser
from parsers.analysis_result import AnalysisResult
from parsers.code_chunk import CodeChunk

logger = logging.getLogger(__name__)


class Analyzer(ABC):
    """Analyseur de code de base retournant des AnalysisResult"""

    language: str
    filename_parser: FilenameParser

    def __init__(self, language: str):
        self.filename_parser = FilenameParser()
        self.language = language

    def analyze_file(self, file_path: str) -> AnalysisResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Analyser le contenu
            result = self.analyze_content(content, file_path)

            # Dans chaque analyseur, après avoir essayé de trouver des chunks spécifiques :

            # À la fin de la méthode analyze_content :
            if not result.chunks and content.strip():
                result.chunks.append(
                    CodeChunk(
                        language=self.language,
                        name="file_content",
                        content=content[:1000],  # Tronquer si trop long
                        file_path=file_path,
                        start_line=1,
                        end_line=len(content.splitlines())
                    )
                )
                result.symbols.append("file_content")

            # Mettre à jour les résultats
            self._update_result(result, Path(file_path))

            return result

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return self._create_error_result(Path(file_path), str(e))

    @abstractmethod
    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        """Analyse du contenu (méthode interne pour compatibilité)"""
        pass

    def _update_result(self, result: AnalysisResult, file_path: Path) -> AnalysisResult:
        """Crée un AnalysisResult de base avec métadonnées"""
        parts = self.filename_parser.parse_filename(str(file_path))
        lm = file_path.stat().st_mtime
        result.file_path = file_path
        result.file_name = file_path.name
        result.last_modified = lm
        result.language = self.language
        result.name_parts = parts["tokens"]
        for chunk in result.chunks:
            chunk.name_parts = parts["tokens"]
        return result

    def _create_error_result(self, file_path: Path, error: str) -> AnalysisResult:
        """Crée un résultat d'erreur"""
        result = AnalysisResult(self.language)
        self._update_result(result, file_path)
        result.errors.append(error)
        return result
