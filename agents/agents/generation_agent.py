from agents.base_agent import BaseAgent
from models.analysis_context import AnalysisContext
from models.code_chunk import CodeChunk
from typing import Dict

from models.project_structure import ProjectStructure


class GenerationAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config, "generation_agent")
        self.config = config

    def generate(self, context: AnalysisContext, analysis_results: Dict[str, ProjectStructure]) -> Dict[str, CodeChunk]:
        # Transformer les résultats d'analyse en texte
        analysis_text = self._analysis_to_text(analysis_results)

        system_prompt = "Vous êtes un assistant développeur. Générez du code ou de la documentation."

        llm_response = self._call_llm(prompt=analysis_text, system_prompt=system_prompt)

        generated_chunk = CodeChunk(
            content=llm_response,
            file_path="generated/main.py",
            filename="main.py",
            language="python",
            category="generated",
            chunk_type="code"
        )

        return {generated_chunk.filename: generated_chunk}

    def _analysis_to_text(self, analysis_results: Dict[str, ProjectStructure]) -> str:
        lines = []
        for agent_name, struct in analysis_results.items():
            lines.append(f"## Résultats de {agent_name}")
            lines.append(f"Résumé: {getattr(struct, 'summary', '')}")
            if hasattr(struct, 'components'):
                lines.append(f"Composants: {', '.join(getattr(struct, 'components', []))}")
            if hasattr(struct, 'patterns'):
                lines.append(f"Patterns: {', '.join(getattr(struct, 'patterns', []))}")
        return "\n\n".join(lines)
