import json

from parsers.analysis_result import AnalysisResult
from parsers.analyzer import Analyzer
from parsers.code_chunk import CodeChunk


class JSONAnalyzer(Analyzer):

    def __init__(self):
        super().__init__("json")

    def analyze_content(self, content: str, file_path: str) -> AnalysisResult:
        result = AnalysisResult(language=self.language)

        try:
            # Parser le JSON
            data = json.loads(content)

            # Si c'est un objet, prendre ses clés comme symboles
            if isinstance(data, dict):
                for key in data.keys():
                    result.symbols.append(key)

                    # Créer un chunk simple pour chaque clé
                    result.chunks.append(
                        CodeChunk(
                            language=self.language,
                            name=key,
                            content=f"{key}",
                            file_path=file_path,
                            start_line=1,
                            end_line=1
                        )
                    )

            # Marquer comme valide
            result.is_valid = True

        except json.JSONDecodeError:
            # JSON invalide - pas de chunks
            result.is_valid = False

        return result
