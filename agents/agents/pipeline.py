# agents/pipeline.py
from typing import Dict
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure

class AnalysisPipeline:
    """
    Orchestrateur d'agents.
    Chaque agent retourne un ProjectStructure ou similaire.
    """

    def __init__(self, agents: list):
        self.agents = agents

    def run(self, context: AnalysisContext) -> Dict[str, ProjectStructure]:
        """
        Retourne un dict {agent_name: ProjectStructure}.
        """
        results = {}
        for agent in self.agents:
            result: ProjectStructure = agent.analyze(context)
            results[agent.agent_name] = result
        return results
