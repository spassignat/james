import logging
from typing import List
from models.analysis_context import AnalysisContext
from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure
from agents.pipeline import AnalysisPipeline
from agents.generation_agent import GenerationAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Orchestration des agents : analyse + génération
    """
    def __init__(self, config):
        self.config = config
        self.pipeline = AnalysisPipeline(config)
        self.generation_agent = GenerationAgent(config)

    def run_analysis_pipeline(self, context: AnalysisContext) -> List[ProjectStructure]:
        """
        Exécute le pipeline d'analyse et retourne les ProjectStructure.
        """
        return self.pipeline.run(context)

    def run_generation(
            self,
            context: AnalysisContext,
            analysis_results: List[ProjectStructure]
    ) -> List[CodeChunk]:
        """
        Exécute le générateur de code à partir des résultats d'analyse.
        """
        return self.generation_agent.generate(context, analysis_results)

    def run_full_pipeline(self, context: AnalysisContext) -> List[CodeChunk]:
        """
        Pipeline complet : analyse puis génération.
        """
        analysis_results = self.run_analysis_pipeline(context)
        generation_results = self.run_generation(context, analysis_results)
        return generation_results
