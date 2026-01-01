# src/main_analysis.py

import logging
from typing import List, Dict, Any

from parsers.utils.Util import infer_language_from_path, infer_category_from_type
from models.analysis_context import AnalysisContext
from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure

from agents.agent_manager import AgentManager
from config.config_loader import ConfigLoader
from main_doc import RuleGenerator
from project_analyzer import ProjectAnalyzer
from vector.vector_store import VectorStore, get_existing_chunks, get_all_chunks_direct

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def analyze_project_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse la structure du projet"""
    logger.info("🔍 Analyse de la structure du projet...")
    project_analyzer = ProjectAnalyzer(config)
    return project_analyzer.analyze_project_structure()


def get_chunks_for_analysis(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Récupère les chunks pour l'analyse"""
    logger.info("📚 Récupération des chunks depuis la base vectorielle...")

    vector_store = VectorStore(config)

    # Méthode 1: Récupération directe (plus efficace)
    chunks = get_all_chunks_direct(vector_store, limit=2000)

    if not chunks:
        # Méthode 2: Fallback avec recherche neutre
        logger.warning(
            "Méthode directe échouée, utilisation de la méthode de recherche..."
        )
        chunks = get_existing_chunks(vector_store, config, limit=1000)

    # Filtrer et organiser les chunks par type pour l'analyse
    organized_chunks = organize_chunks_by_type(chunks)

    return organized_chunks


def organize_chunks_by_type(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Organise les chunks par type pour une analyse plus efficace"""
    organized = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type", "unknown")
        file_path = metadata.get("file_path", "")

        enhanced_chunk = {
            "content": chunk["content"],
            "type": chunk_type,
            "file_path": file_path,
            "filename": metadata.get("filename", ""),
            "language": infer_language_from_path(file_path),
            "category": infer_category_from_type(chunk_type, file_path),
        }
        organized.append(enhanced_chunk)

    return organized


def main():
    """Point d'entrée principal pour l'analyse rétrospective"""
    try:
        # Chargement configuration
        logger.info("⚙️ Chargement de la configuration...")
        config_loader = ConfigLoader()
        config = config_loader.config

        # Étape 1: Analyse de la structure du projet
        raw_structure = analyze_project_structure(config)

        # Étape 2: Récupération des chunks existants
        raw_chunks = get_chunks_for_analysis(config)

        if not raw_chunks:
            logger.error("❌ Aucun chunk trouvé dans la base vectorielle")
            return

        # Étape 2b: Création des objets CodeChunk
        chunks = [
            CodeChunk(
                content=c["content"],
                file_path=c["file_path"],
                filename=c["filename"],
                language=c["language"],
                category=c["category"],
                chunk_type=c["type"],
            )
            for c in raw_chunks
        ]

        # Étape 3: Création de l'objet AnalysisContext
        analysis_context = AnalysisContext(
            project_structure=ProjectStructure(**raw_structure),
            chunks=chunks,
            config=config,
        )

        logger.info(f"📊 Analyse basée sur {len(chunks)} chunks et structure de projet")

        # Étape 4: Exécution du pipeline d'agents
        logger.info("🤖 Lancement des agents d'analyse...")
        agent_manager = AgentManager(config_loader)
        results = agent_manager.run_analysis_pipeline(analysis_context)

        # Étape 5: Génération de la documentation
        logger.info("📝 Génération de la documentation...")
        rule_generator = RuleGenerator(config)
        documentation_path = rule_generator.generate_rules_documentation(results)

        logger.info(f"✅ Analyse terminée! Documentation générée: {documentation_path}")

        # Résumé
        stats = {
            "total_chunks_analyzed": len(chunks),
            "patterns_identified": len(raw_structure.get("patterns", {})),
            "documentation_path": documentation_path,
        }

        logger.info(f"📈 Résumé: {stats}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}")
        raise


if __name__ == "__main__":
    main()
