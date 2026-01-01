# src/agents/agent_manager.py (version corrigée)

import logging
from typing import List, Dict, Any

from agents.architecture_agent import ArchitectureAgent
from agents.base_agent import BaseAgent
from agents.generative_description_agent import GenerativeDescriptionAgent
from agents.pattern_agent import PatternAgent
from agents.rules_agent import RulesAgent

logger = logging.getLogger(__name__)

# Dans BaseAgent.__init__


class AgentManager:
    def __init__(self, config_loader):
        self.config = config_loader.config
        self.agents = self._initialize_agents()
        self.analysis_history = []

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialise tous les agents"""
        agents = {}

        try:
            agents['architecture'] = ArchitectureAgent(self.config)
            logger.info("✅ ArchitectureAgent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation ArchitectureAgent: {e}")
            agents['architecture'] = None

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

        try:
            agents['generative'] = GenerativeDescriptionAgent(self.config)
            logger.info("✅ GenerativeDescriptionAgent initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation GenerativeDescriptionAgent: {e}")
            agents['generative'] = None

        return agents

    def run_analysis_pipeline(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le pipeline complet d'analyse"""
        logger.info("🚀 Démarrage du pipeline d'analyse des agents...")

        results = {
            'pipeline_start': self._get_timestamp(),
            'status': 'running',
            'agents_executed': [],
            'architecture': {},
            'errors': []
        }

        try:
            # Vérifier que le contexte contient les données nécessaires
            required_keys = ['project_structure', 'chunks', 'file_patterns']
            missing_keys = [key for key in required_keys if key not in analysis_context]

            if missing_keys:
                error_msg = f"Clés manquantes dans le contexte: {missing_keys}"
                logger.error(f"❌ {error_msg}")
                results['errors'].append(error_msg)
                results['status'] = 'failed'
                return results

            # Étape 1: Analyse d'architecture
            if self.agents['architecture']:
                logger.info("🔧 Exécution de l'ArchitectureAgent...")
                try:
                    architecture_result = self.agents['architecture'].analyze(analysis_context)
                    results['architecture'] = architecture_result
                    results['agents_executed'].append('architecture')
                except Exception as e:
                    error_msg = f"Erreur ArchitectureAgent: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    architecture_result = {'content': 'Analyse échouée', 'summary': 'Échec'}
            else:
                architecture_result = {'content': 'Agent non disponible', 'summary': 'Non exécuté'}
                results['errors'].append('ArchitectureAgent non initialisé')

            # Étape 2: Analyse des patterns
            if self.agents['patterns']:
                logger.info("🔍 Exécution du PatternAgent...")
                try:
                    pattern_result = self.agents['patterns'].analyze(analysis_context)
                    results['patterns'] = pattern_result
                    results['agents_executed'].append('patterns')
                except Exception as e:
                    error_msg = f"Erreur PatternAgent: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    pattern_result = {'content': 'Analyse échouée', 'patterns_identified': []}
            else:
                pattern_result = {'content': 'Agent non disponible', 'patterns_identified': []}
                results['errors'].append('PatternAgent non initialisé')

            # Étape 2: Analyse des patterns
            if self.agents['generative']:
                logger.info("🔍 Exécution du GenerativeAgent...")
                try:
                    pattern_result = self.agents['generative'].analyze(analysis_context)
                    results['generative'] = pattern_result
                    results['agents_executed'].append('generative')
                except Exception as e:
                    error_msg = f"Erreur GenerativeAgent: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    pattern_result = {'content': 'Analyse échouée', 'generative_identified': []}
            else:
                pattern_result = {'content': 'Agent non disponible', 'generative_identified': []}
                results['errors'].append('GenerativeAgent non initialisé')

            # Étape 3: Génération des règles
            if self.agents['rules']:
                logger.info("📋 Exécution du RulesAgent...")
                try:
                    rules_result = self.agents['rules'].analyze({
                        'architecture_analysis': architecture_result,
                        'pattern_analysis': pattern_result
                    })
                    results['rules'] = rules_result
                    results['agents_executed'].append('rules')
                except Exception as e:
                    error_msg = f"Erreur RulesAgent: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
                    rules_result = {'content': 'Génération échouée', 'rules': []}
            else:
                rules_result = {'content': 'Agent non disponible', 'rules': []}
                results['errors'].append('RulesAgent non initialisé')

            # Résumé final
            results['pipeline_end'] = self._get_timestamp()
            results['status'] = 'completed' if not results['errors'] else 'completed_with_errors'
            results['summary'] = self._generate_summary(results)

            # Sauvegarder dans l'historique
            self.analysis_history.append(results)

            logger.info(f"✅ Pipeline d'analyse terminé. Statut: {results['status']}")
            logger.info(f"   Agents exécutés: {len(results['agents_executed'])}")
            logger.info(f"   Erreurs: {len(results['errors'])}")

        except Exception as e:
            logger.error(f"❌ Erreur critique dans le pipeline: {e}")
            results['status'] = 'failed'
            results['errors'].append(f"Erreur critique: {e}")

        return results

    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Génère un résumé des résultats"""
        summary = {
            'total_agents': 3,
            'executed_agents': len(results.get('agents_executed', [])),
            'total_errors': len(results.get('errors', [])),
            'has_architecture': 'architecture' in results,
            'has_patterns': 'patterns' in results,
            'has_rules': 'rules' in results
        }

        # Compter les règles générées
        if 'rules' in results:
            rules_data = results['rules']
            summary['rules_generated'] = rules_data.get('rules_count', 0)

        # Compter les patterns identifiés
        if 'patterns' in results:
            patterns_data = results['patterns']
            summary['patterns_identified'] = patterns_data.get('patterns_count', 0)

        return summary

    def _get_timestamp(self) -> str:
        """Retourne un timestamp formaté"""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_agent_status(self) -> Dict[str, bool]:
        """Retourne le statut de chaque agent"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent is not None and hasattr(agent, 'base_url') and agent.base_url is not None
        return status

    def get_analysis_history(self) -> List[Dict]:
        """Retourne l'historique des analyses"""
        return self.analysis_history

    def clear_history(self):
        """Vide l'historique des analyses"""
        self.analysis_history = []
        logger.info("🗑️  Historique des analyses vidé")
