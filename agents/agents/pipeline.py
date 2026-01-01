import logging
from typing import List
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure
from agents.architecture_agent import ArchitectureAgent

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """
    Pipeline pour exécuter tous les agents d'analyse séquentiellement.
    """
    def __init__(self, config):
        self.config = config
        self.architecture_agent = ArchitectureAgent(config)

    def run(self, context: AnalysisContext) -> List[ProjectStructure]:
        """
        Retourne la liste des ProjectStructure produites par les agents.
        """
        results: List[ProjectStructure] = []

        # Architecture analysis
        arch_result = self.architecture_agent.analyze(context)
        results.append(arch_result)

        # Ici tu pourrais ajouter d'autres agents d'analyse

        logger.info(f"📊 Pipeline terminé: {len(results)} résultats")
        return results
