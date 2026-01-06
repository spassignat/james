from typing import Dict, Any

from agents.analysis_context import AnalysisContext
from agents.generic_agent import GenericAgent
from models.search_intent import SearchIntent


class ArchitectureAgent(GenericAgent):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "architecture_agent")

    def get_intent(self, context: AnalysisContext) -> SearchIntent:
        return SearchIntent(
            goal="Comprendre l’architecture globale du projet",
            domain="architecture",
            focus=["project_structure", "agents", "analysis_pipeline", "vector_store"],
            depth="medium"
        )

    def get_system_prompt(self) -> str:
        return "Vous êtes un architecte logiciel expert. Fournissez une structure d'architecture complète."
