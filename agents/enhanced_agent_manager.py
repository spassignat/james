import logging
from typing import Dict, Any, List

from agents.enhanced_architecture_agent import EnhancedArchitectureAgent
from agents.pattern_agent import PatternAgent
from agents.rules_agent import RulesAgent

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader
from project_analyzer import ProjectAnalyzer

# Configuration du logging
logger = logging.getLogger(__name__)


class EnhancedAgentManager:
    """Manager d'agents amélioré avec analyse de projet"""

    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.config
        self.project_analyzer = ProjectAnalyzer(self.config)
        self.agents = self._initialize_enhanced_agents()
        self.analysis_history = []

    def _initialize_enhanced_agents(self) -> Dict[str, BaseAgent]:
        """Initialise les agents améliorés"""
        agents = {}

        try:
            agents['architecture'] = EnhancedArchitectureAgent(self.config)
            logger.info("✅ EnhancedArchitectureAgent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation EnhancedArchitectureAgent: {e}")
            agents['architecture'] = None

        # Garder les autres agents (PatternAgent, RulesAgent) inchangés
        try:
            agents['patterns'] = PatternAgent(self.config)
            logger.info("✅ PatternAgent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation PatternAgent: {e}")
            agents['patterns'] = None

        try:
            agents['rules'] = RulesAgent(self.config)
            logger.info("✅ RulesAgent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation RulesAgent: {e}")
            agents['rules'] = None

        return agents

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyse la structure du projet"""
        return self.project_analyzer.analyze_project_structure()

    def run_enhanced_pipeline(self, chunks: List[Dict] = None) -> Dict[str, Any]:
        """Exécute le pipeline amélioré avec analyse de structure automatique"""
        logger.info("🚀 Démarrage du pipeline amélioré avec analyse de structure...")

        # Étape 0: Analyser la structure du projet
        logger.info("📊 Analyse de la structure du projet...")
        project_structure = self.analyze_project_structure()

        # Étape 1: Analyser les patterns de fichiers
        file_patterns = project_structure.get('file_patterns', {})

        # Construire le contexte d'analyse
        analysis_context = {
            'project_structure': project_structure,
            'chunks': chunks or [],
            'file_patterns': file_patterns,
            'project_stats': {
                'total_files': project_structure['total_files'],
                'extensions': project_structure['extensions_found']
            }
        }

        # Exécuter le pipeline standard
        return self.run_analysis_pipeline(analysis_context)

    # Méthodes héritées de AgentManager
    def run_analysis_pipeline(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Hérite de la méthode originale (à adapter selon votre implémentation)"""
        # ... (le code existant de run_analysis_pipeline)
        pass

    def get_agent_status(self) -> Dict[str, bool]:
        """Retourne le statut de chaque agent"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent is not None and hasattr(agent, 'ollama_client') and agent.ollama_client is not None
        return status
