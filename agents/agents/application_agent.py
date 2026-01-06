from typing import Dict, Any

from agents.analysis_context import AnalysisContext
from agents.generic_agent import GenericAgent
from models.search_intent import SearchIntent


class ApplicationAgent(GenericAgent):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "application_agent")

    def get_intent(self, context: AnalysisContext) -> SearchIntent:
        return SearchIntent(
            goal="Analyser l’architecture applicative : patterns et flux de données",
            domain="analysis",
            focus=["services", "repositories", "models", "data_flow", "business_logic"],
            depth="medium"
        )

    def get_system_prompt(self) -> str:
        return "Vous êtes un expert en architecture applicative. Analysez les patterns, la structure des données et la logique applicative."
