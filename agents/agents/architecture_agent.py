from typing import List, Optional
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure
from agents.base_agent import BaseAgent

class ArchitectureAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__(config, "architecture_agent")

    def analyze(self, context: AnalysisContext) -> ProjectStructure:
        # Transformer les chunks en texte pour l'IA
        chunks_text = self._chunks_to_text(context.chunks)

        system_prompt = (
            "Vous êtes un architecte logiciel expert. "
            "Analysez le projet et fournissez une structure d'architecture."
        )

        llm_response = self._call_llm(prompt=chunks_text, system_prompt=system_prompt)

        # TODO: parser la réponse pour créer un ProjectStructure
        project_structure = ProjectStructure(
            summary=llm_response,
            components=[],  # remplir si besoin
            patterns=[]
        )
        return project_structure

    def _chunks_to_text(self, chunks: List) -> str:
        """
        Transforme les CodeChunk en texte pour le prompt.
        Limite la taille et met en forme.
        """
        lines = []
        for chunk in chunks[:10]:  # max 10 chunks
            lines.append(f"### {chunk.filename} ({chunk.language}, {chunk.category})")
            content_preview = chunk.content[:300].replace("```", "'``'")
            lines.append(f"```\n{content_preview}\n```")
        return "\n\n".join(lines)
