# src/main_doc/rule_generator.py
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure

logger = logging.getLogger(__name__)


class RuleGenerator:
    """
    Générateur de documentation et de règles à partir des résultats d'analyse.
    Fonctionne avec ProjectStructure et CodeChunk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "./docs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_rules_documentation(
            self, analysis_results: List[ProjectStructure], generated_chunks: List[CodeChunk] = None
    ) -> str:
        """
        Génère un document Markdown récapitulatif des analyses et du code généré.

        Args:
            analysis_results: Liste des ProjectStructure issues de l'analyse.
            generated_chunks: Liste optionnelle des CodeChunk générés pour le projet.

        Returns:
            Le chemin complet vers le fichier Markdown généré.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_path = self.output_dir / f"project_rules_{timestamp}.md"

        try:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(f"# Documentation de projet - {timestamp}\n\n")

                # Résumé global
                f.write("## Résumé de l'architecture et des patterns\n")
                for ps in analysis_results:
                    f.write(f"- **Projet**: {ps.project_name}\n")
                    f.write(f"  - Total modules: {len(ps.modules)}\n")
                    f.write(f"  - Patterns identifiés: {', '.join(ps.patterns_identified or [])}\n\n")

                # Code généré
                if generated_chunks:
                    f.write("## Code généré / Suggestions\n")
                    for chunk in generated_chunks:
                        f.write(f"### {chunk.filename} ({chunk.language})\n")
                        f.write(f"```{chunk.language}\n{chunk.content}\n```\n\n")

            logger.info(f"✅ Documentation générée: {doc_path}")
            return str(doc_path)

        except Exception as e:
            logger.error(f"❌ Erreur génération documentation: {e}")
            return ""
