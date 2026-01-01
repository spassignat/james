# agents/agent_manager.py
from typing import Dict

from agents.architecture_agent import ArchitectureAgent
from agents.generation_agent import GenerationAgent
from agents.pipeline import AnalysisPipeline
from models.analysis_context import AnalysisContext
from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure

class AgentManager:
    def __init__(self, config_loader):
        self.config = config_loader.config
        self.agents = [ArchitectureAgent(self.config)]
        self.generation_agent = GenerationAgent(self.config)
        self.pipeline = AnalysisPipeline(self.agents)

    def run_analysis_pipeline(self, context: AnalysisContext) -> Dict[str, object]:
        """
        Retourne un dict :
        {
            "analysis": Dict[agent_name, ProjectStructure],
            "generation": Dict[filename, CodeChunk]
        }
        """
        analysis_results: Dict[str, ProjectStructure] = self.pipeline.run(context)
        generation_results: Dict[str, CodeChunk] = self.generation_agent.generate(context, analysis_results)

        return {
            "analysis": analysis_results,
            "generation": generation_results
        }
