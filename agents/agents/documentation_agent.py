import logging
from typing import List

from agents.analysis_context import AnalysisContext
from agents.base_agent import BaseAgent
from models.project_structure import ProjectStructure
from parsers.code_chunk import CodeChunk

logger = logging.getLogger(__name__)


class DocumentationAgent(BaseAgent):
    """
    Génère du code ou de la documentation à partir des résultats d'analyse.
    """

    def __init__(self, config):
        super().__init__(config, agent_name="documentation_agent")

    def generate(
            self,
            context: AnalysisContext,
            analysis_results: List[ProjectStructure]
    ) -> List[CodeChunk]:
        """
        Génère des CodeChunk à partir des ProjectStructure.
        """
        generated_chunks = []

        # Logique d'exemple : génération fictive
        for i, structure in enumerate(analysis_results):
            chunk = CodeChunk(
                content=f"# Documentation générée pour structure {i}",
                file_path=f"/generated/generated_doc_{i}.md",
                language="markdown",
                chunk_type="generated_doc"
            )
            generated_chunks.append(chunk)

        logger.info(f"✅ {len(generated_chunks)} chunks générés")
        return generated_chunks
