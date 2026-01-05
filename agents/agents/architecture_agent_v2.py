from typing import List, Tuple, Dict, Any
from overrides import overrides

from agents.base_agent import BaseAgent
from agents.generic_agent import GenericAgent
from models.analysis_context import AnalysisContext
from models.project_structure import ProjectStructure
from models.search_intent import SearchIntent
from vector.vector_store import VectorStore

ARCHITECTURE_INTENTS: List[SearchIntent] = [
    SearchIntent(
        goal="Identifier la structure globale du projet",
        domain="architecture",
        focus=["folders", "modules", "project_structure"],
        depth="high",
    ),
    SearchIntent(
        goal="Identifier les composants principaux et leurs responsabilités",
        domain="architecture",
        focus=["components", "services", "controllers", "agents"],
        depth="high",
    ),
    SearchIntent(
        goal="Identifier les frameworks, bibliothèques et dépendances",
        domain="architecture",
        focus=["frameworks", "dependencies", "config"],
        depth="high",
    ),
    SearchIntent(
        goal="Identifier les points d’entrée et flux principaux",
        domain="architecture",
        focus=["entrypoints", "main", "routing"],
        depth="high",
    ),
]

class ArchitectureAgentV2(GenericAgent):
    """
    Analyse l’architecture technique globale d’un projet.
    - multi-intents
    - agrégation
    - déduplication
    - diversification
    """

    def __init__(self, config: dict):
        super().__init__(config, "architecture_agent")

    @overrides
    def analyze(
            self,
            context: AnalysisContext,
            vector_store: VectorStore
    ) -> ProjectStructure:

        aggregated_chunks: List[Tuple[str, Dict[str, Any]]] = []

        # 1️⃣ Lancer plusieurs recherches ciblées
        for intent in ARCHITECTURE_INTENTS:
            results = vector_store.search(intent)
            aggregated_chunks.extend(self._extract_chunks(results))

        # 2️⃣ Déduplication logique
        unique_chunks = self._deduplicate_chunks(aggregated_chunks)

        # 3️⃣ Forcer la diversité (1 chunk max par fichier)
        diversified_chunks = self._limit_one_chunk_per_file(unique_chunks)

        # 4️⃣ Préparer le prompt
        chunks_text = self._chunks_to_text(diversified_chunks)

        system_prompt = (
            "Vous êtes un architecte logiciel senior.\n"
            "À partir des extraits fournis, décrivez clairement :\n"
            "- la structure globale du projet\n"
            "- les grandes couches techniques\n"
            "- les composants majeurs et leurs responsabilités\n"
            "- les frameworks et technologies utilisés\n"
            "- les flux principaux\n"
            "Soyez synthétique, structuré et précis."
        )

        llm_response = self._call_llm(
            prompt=chunks_text,
            system_prompt=system_prompt
        )

        return ProjectStructure(
            project_name=context.project_name,
            architecture_overview=llm_response,
            extra_metadata={
                "agent": "architecture_agent_v2",
                "intents_used": len(ARCHITECTURE_INTENTS),
                "chunks_used": len(diversified_chunks),
            }
        )
    def _extract_chunks(
            self,
            results: dict
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        ChromaDB renvoie:
        - documents: List[List[str]]
        - metadatas: List[List[dict]]
        """
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return list(zip(documents, metadatas))
    def _deduplicate_chunks(
            self,
            chunks: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        seen = set()
        unique = []

        for content, meta in chunks:
            chunk_id = meta.get("chunk_id") or hash(content)
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append((content, meta))

        return unique
    def _limit_one_chunk_per_file(
            self,
            chunks: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        seen_files = set()
        diversified = []

        for content, meta in chunks:
            file_path = meta.get("relative_path") or meta.get("file_path")
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                diversified.append((content, meta))

        return diversified
    def _chunks_to_text(
            self,
            chunks: List[Tuple[str, Dict[str, Any]]],
            max_chars_per_chunk: int = 400
    ) -> str:
        lines = []

        for content, meta in chunks:
            filename = meta.get("filename", "unknown")
            language = meta.get("language", "unknown")
            chunk_type = meta.get("chunk_type", "unknown")

            preview = content[:max_chars_per_chunk].replace("```", "'``'")

            lines.append(
                f"### {filename} ({language}, {chunk_type})\n"
                f"```\n{preview}\n```"
            )

        return "\n\n".join(lines)
