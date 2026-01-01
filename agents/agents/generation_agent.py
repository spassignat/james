import logging
from typing import List
from models.analysis_context import AnalysisContext
from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class GenerationAgent(BaseAgent):
    """
    Génère du code ou de la documentation à partir des résultats d'analyse.
    """
    def __init__(self, config):
        super().__init__(config, agent_name="generation_agent")

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
                filename=f"generated_doc_{i}.md",
                file_path=f"/generated/generated_doc_{i}.md",
                language="markdown",
                category="documentation",
                chunk_type="generated_doc"
            )
            generated_chunks.append(chunk)

        logger.info(f"✅ {len(generated_chunks)} chunks générés")
        return generated_chunks
