from typing import Dict, Any

from agents.analysis_context import AnalysisContext
from agents.generic_agent import GenericAgent
from models.search_intent import SearchIntent


class SecurityAgent(GenericAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "security_agent")

    def get_intent(self, context: AnalysisContext) -> SearchIntent:
        return SearchIntent(
            goal="Analyser la sécurité et la conformité du projet",
            domain="security",
            focus=["auth", "authorization", "encryption", "sensitive_data", "config"],
            depth="low"
        )

    def get_system_prompt(self) -> str:
        return "Vous êtes un expert en sécurité logicielle. Analysez les vulnérabilités et la gestion des données sensibles."
