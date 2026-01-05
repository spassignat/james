# src/main_analysis.py
import json
import logging
from config.config_loader import ConfigLoader
from project_analyzer import ProjectAnalyzer
from vector.vector_store import VectorStore
from agents.agent_manager import AgentManager
from models.analysis_context import AnalysisContext

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Point d'entrée principal pour l'analyse et la génération de code"""
    try:
        # Chargement configuration
        logger.info("⚙️  Chargement de la configuration...")
        config_loader = ConfigLoader()
        config = config_loader.config

        # Étape 1: Analyse de la structure du projet
        logger.info("🔍 Analyse de la structure du projet...")
        project_analyzer = ProjectAnalyzer(config)
        project_structure = project_analyzer.analyze_project_structure()

        # Étape 2: Récupération des chunks vectorisés
        logger.info("📚 Récupération des chunks depuis la base vectorielle...")
        vector_store = VectorStore(config)
        chunks = vector_store.get_all_chunks(limit=2000)

        if not chunks:
            logger.error("❌ Aucun chunk trouvé dans la base vectorielle")
            return

        logger.info(f"📊 Analyse basée sur {len(chunks)} chunks et structure de projet")

        # Étape 3: Préparer le contexte d'analyse
        context = AnalysisContext(
            project_structure=project_structure,
            chunks=chunks,
            project_config=config.get('project', {})
        )

        # Étape 4: Exécution du pipeline d'agents d'analyse
        logger.info("🤖 Lancement des agents d'analyse...")
        agent_manager = AgentManager(config)
        analysis_results = agent_manager.run_analysis_pipeline(context,vector_store)

        # On peut ici envisager un pipeline de génération plus tard
        # generation_results = generation_agent.generate(context, analysis_results)

        # Étape 5: Sauvegarde ou export des résultats
        logger.info("💾 Sauvegarde des résultats...")
        vector_store.persist_index()  # sauvegarde de l'index et persistance
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=4)

        # Étape 6: Résumé
        stats = {
            'total_chunks': len(chunks),
            'total_modules': len(project_structure.modules),
            'patterns_identified': project_structure.patterns_identified,
            'analysis_count': len(analysis_results),
        }
        logger.info(f"📈 Résumé de l'analyse: {stats}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
