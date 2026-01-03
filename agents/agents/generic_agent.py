from typing import Dict, Any

from agents.base_agent import BaseAgent
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure
from models.search_intent import SearchIntent
from vector.vector_store import VectorStore


class GenericAgent(BaseAgent):
    """
    Agent générique pour JAMES.
    Chaque sous-classe définit son SearchIntent et system_prompt.
    """

    def __init__(self, config: Dict[str, Any], agent_name: str):
        super().__init__(config, agent_name)

    def analyze(self, context: AnalysisContext, vector_store: VectorStore) -> ProjectStructure:
        # Générer l'intent
        intent = self.get_intent(context)

        # Récupérer les chunks pertinents
        chunks = vector_store.search(
            intent,
            top_k=self.select_top_k(intent)
        )

        # Transformer les chunks en texte
        chunks_text = self._chunks_to_text(chunks)

        # Appeler le LLM
        llm_response = self._call_llm(prompt=chunks_text, system_prompt=self.get_system_prompt())

        # Parser la réponse pour créer le ProjectStructure
        project_structure = self.parse_llm_response(llm_response, context)

        return project_structure

    def _chunks_to_text(self, search_results: dict, max_chunks: int = 10000) -> str:
        """
        Transforme les résultats de ChromaDB en texte pour le prompt LLM.
        :param search_results: dict renvoyé par collection.query()
        :param max_chunks: limite le nombre de chunks utilisés
        """
        lines = []

        # Chroma renvoie 'documents' comme List[List[str]]
        documents = search_results.get("documents", [])

        # documents peut être une liste de listes
        flat_docs = [doc for sublist in documents for doc in sublist]

        for idx, content in enumerate(flat_docs[:max_chunks]):
            lines.append(f"### Chunk {idx + 1}")
            content_preview = content[:300].replace("```", "'``'")  # limiter la taille
            lines.append(f"```\n{content_preview}\n```")

        return "\n\n".join(lines)

    def select_top_k(self, intent: SearchIntent) -> int:
        return 8 if intent.depth == "high" else 4 if intent.depth == "medium" else 3

    # À surcharger par chaque agent
    def get_intent(self, context: AnalysisContext) -> SearchIntent:
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        raise NotImplementedError

    def parse_llm_response(self, response: str, context: AnalysisContext) -> ProjectStructure:
        # Par défaut, on met le résumé dans architecture_overview
        return ProjectStructure(
            project_name=context.project_name,
            architecture_overview=response
        )
