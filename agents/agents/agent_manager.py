import logging
from typing import List

from agents.pipeline import AnalysisPipeline
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure
from vector.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Orchestration des agents : analyse + génération
    """

    def __init__(self, config):
        self.config = config
        self.pipeline = AnalysisPipeline(config)

    def run_analysis_pipeline(self, context: AnalysisContext, vector_store: VectorStore) -> List[ProjectStructure]:
        """
        Exécute le pipeline d'analyse et retourne les ProjectStructure.
        """
        return self.pipeline.run(context, vector_store)
