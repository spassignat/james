from typing import Dict, Any

from agents.generic_agent import GenericAgent
from models.analysis_context import AnalysisContext
from models.search_intent import SearchIntent


class FunctionalAgent(GenericAgent):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "functional_agent")

    def get_intent(self, context: AnalysisContext) -> SearchIntent:
        return SearchIntent(
            goal="Analyser l’architecture fonctionnelle : workflows et cas d’usage",
            domain="analysis",
            focus=["use_cases", "workflows", "features", "business_logic"],
            depth="medium"
        )

    def get_system_prompt(self) -> str:
        return "Vous êtes un expert en architecture fonctionnelle. Analysez les workflows et la logique métier."
