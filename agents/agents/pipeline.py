import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Type, Dict, Any

from agents.analysis_context import AnalysisContext
from agents.application_agent import ApplicationAgent
from agents.architecture_agent_v2 import ArchitectureAgentV2
from agents.documentation_pipeline import DocumentationPipeline
from agents.functional_agent import FunctionalAgent
from agents.security_agent import SecurityAgent
from models.project_structure import ProjectStructure
from vector.vector_store import VectorStore

# Ajouter ici d'autres agents si besoin

# Import du générateur de documentation

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Pipeline pour exécuter tous les agents d'analyse et générer la documentation.
    """

    def __init__(self, config):
        self.config = config

        # Liste des agents à exécuter
        self.agents: List[Type] = [
            ArchitectureAgentV2,
            ApplicationAgent,
            FunctionalAgent,
            SecurityAgent
        ]

        # Initialisation des instances d'agents
        self.agent_instances = [agent_class(config) for agent_class in self.agents]

        # Pipeline de documentation
        doc_config = config.get("documentation", {
            "template_dir": "templates",
            "output_dir": "documentation"
        })
        self.doc_pipeline = DocumentationPipeline(
            template_dir=doc_config.get("template_dir", "templates"),
            output_dir=doc_config.get("output_dir", "documentation"),
            custom_variables=doc_config.get("custom_variables", {})
        )

        self.results: List[ProjectStructure] = []
        self.merged_structure: ProjectStructure = None
        self.generated_docs: Dict[str, Any] = {}

    def run(self, context: AnalysisContext, vector_store: VectorStore) -> Dict[str, Any]:
        """
        Exécute tous les agents, fusionne les résultats et génère la documentation.
        """
        logger.info("🚀 Démarrage du pipeline d'analyse complet")

        # 1. Exécuter tous les agents
        agent_results = self._run_agents(context, vector_store)

        # 2. Fusionner les résultats des agents
        merged_structure = self._merge_results(agent_results)

        # 3. Générer la documentation
        documentation = self._generate_documentation(merged_structure)

        # 4. Sauvegarder les résultats
        self._save_results(agent_results, merged_structure, documentation)

        return {
            "agent_results": agent_results,
            "merged_structure": merged_structure,
            "documentation": documentation,
            "summary": self._generate_summary()
        }

    def _run_agents(self, context: AnalysisContext, vector_store: VectorStore) -> List[ProjectStructure]:
        """Exécute tous les agents"""
        results: List[ProjectStructure] = []

        for agent in self.agent_instances:
            logger.info(f"🚀 Lancement de l'agent {agent.agent_name}")
            try:
                result = agent.analyze(context, vector_store)
                results.append(result)
                logger.info(f"✅ Agent {agent.agent_name} terminé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur dans l'agent {agent.agent_name}: {e}")
                # Créer un résultat vide pour cet agent
                empty_result = ProjectStructure(
                    project_name=context.project_name,
                    architecture_overview=f"Erreur dans l'agent {agent.agent_name}",
                    extra_metadata={"agent": agent.agent_name, "error": str(e)}
                )
                results.append(empty_result)

        self.results = results
        logger.info(f"📊 Agents terminés: {len(results)} résultats")
        return results

    def _merge_results(self, agent_results: List[ProjectStructure]) -> ProjectStructure:
        """Fusionne les résultats de tous les agents"""
        if not agent_results:
            raise ValueError("Aucun résultat d'agent à fusionner")

        logger.info("🔄 Fusion des résultats des agents...")

        # Prendre le premier résultat comme base
        base_result = agent_results[0]

        merged = ProjectStructure(
            project_name=base_result.project_name,
            architecture_overview=base_result.architecture_overview,
            dependencies=base_result.dependencies.copy(),
            patterns_identified=base_result.patterns_identified.copy(),
            components=base_result.components.copy(),
            modules=base_result.modules.copy(),
            files=base_result.files.copy(),
            extra_metadata={
                "agents_used": [type(agent).__name__ for agent in self.agent_instances],
                "merged_at": datetime.now().isoformat(),
                "agent_details": []
            }
        )

        # Fusionner les autres résultats
        for i, result in enumerate(agent_results):
            # Sauvegarder les métadonnées de chaque agent
            merged.extra_metadata["agent_details"].append({
                "agent": self.agent_instances[i].agent_name if i < len(self.agent_instances) else f"Agent_{i}",
                "analysis_time": result.last_analysis_at,
                "modules_count": len(result.modules),
                "files_count": len(result.files)
            })

            # Fusionner les données
            self._merge_lists(merged.files, result.files)
            self._merge_lists(merged.modules, result.modules)
            self._merge_lists(merged.patterns_identified, result.patterns_identified)
            self._merge_lists(merged.dependencies, result.dependencies)

            # Fusionner les composants
            for comp_type, comp_names in result.components.items():
                for comp_name in comp_names:
                    merged.add_component(comp_type, comp_name)

            # Fusionner l'overview d'architecture
            if result.architecture_overview and result.architecture_overview != base_result.architecture_overview:
                merged.architecture_overview += f"\n\n{result.architecture_overview}"

        # Mettre à jour la date d'analyse
        merged.update_last_analysis()

        self.merged_structure = merged
        logger.info(f"✅ Fusion terminée: {len(merged.files)} fichiers, {len(merged.modules)} modules")

        return merged

    def _merge_lists(self, target: List, source: List):
        """Fusionne deux listes en évitant les doublons"""
        for item in source:
            if item not in target:
                target.append(item)

    def _generate_documentation(self, merged_structure: ProjectStructure) -> Dict[str, Any]:
        """Génère la documentation à partir de la structure fusionnée"""
        logger.info("📝 Génération de la documentation...")

        # Variables personnalisées basées sur la configuration
        custom_vars = {
            "GENERATION_DATE": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "ANALYSIS_METHOD": "Analyse multi-agents IA",
            "AGENTS_COUNT": len(self.agent_instances),
            "ANALYSIS_DURATION": "Quelques secondes",  # À calculer si vous chronométrez
            "CONFIDENCE_SCORE": self._calculate_confidence_score()
        }

        # Mettre à jour les variables personnalisées
        self.doc_pipeline.custom_variables.update(custom_vars)

        # Générer la documentation
        docs = self.doc_pipeline.generate_from_structure(merged_structure)

        self.generated_docs = docs
        logger.info(f"✅ Documentation générée: {len(docs)} fichiers")

        return docs

    def _calculate_confidence_score(self) -> str:
        """Calcule un score de confiance basé sur les résultats"""
        if not self.results:
            return "Faible"

        valid_results = len([r for r in self.results if len(r.files) > 0])
        confidence = (valid_results / len(self.results)) * 100

        if confidence >= 80:
            return "Élevée"
        elif confidence >= 50:
            return "Moyenne"
        else:
            return "Faible"

    def _save_results(self, agent_results: List[ProjectStructure],
                      merged_structure: ProjectStructure,
                      documentation: Dict[str, Any]):
        """Sauvegarde tous les résultats"""
        logger.info("💾 Sauvegarde des résultats...")

        output_dir = self.config.get("output_dir", "output")
        Path(output_dir).mkdir(exist_ok=True)

        # 1. Sauvegarder les résultats individuels des agents
        for i, result in enumerate(agent_results):
            agent_name = self.agent_instances[i].agent_name if i < len(self.agent_instances) else f"Agent_{i}"
            filename = f"{output_dir}/agent_{agent_name.lower().replace(' ', '_')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # 2. Sauvegarder la structure fusionnée
        with open(f"{output_dir}/merged_structure.json", "w", encoding="utf-8") as f:
            json.dump(merged_structure.to_dict(), f, indent=2, ensure_ascii=False)

        # 3. Sauvegarder un rapport de génération
        report = self._generate_summary()
        with open(f"{output_dir}/generation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Résultats sauvegardés dans: {output_dir}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Génère un résumé complet de l'analyse"""
        return {
            "timestamp": datetime.now().isoformat(),
            "project_name": self.merged_structure.project_name if self.merged_structure else "Inconnu",
            "agents_executed": [agent.agent_name for agent in self.agent_instances],
            "agents_successful": len(self.results),
            "documentation_generated": len(self.generated_docs) if self.generated_docs else 0,
            "statistics": {
                "total_files": len(self.merged_structure.files) if self.merged_structure else 0,
                "total_modules": len(self.merged_structure.modules) if self.merged_structure else 0,
                "total_patterns": len(self.merged_structure.patterns_identified) if self.merged_structure else 0,
                "total_dependencies": len(self.merged_structure.dependencies) if self.merged_structure else 0,
                "components_by_type": {
                    comp_type: len(comp_names)
                    for comp_type, comp_names in (self.merged_structure.components.items()
                                                  if self.merged_structure else {})
                }
            },
            "output_locations": {
                "documentation": str(Path(self.doc_pipeline.generator.output_dir).absolute()),
                "analysis_results": self.config.get("output_dir", "output")
            }
        }

    def run_simple(self, context: AnalysisContext, vector_store: VectorStore) -> List[ProjectStructure]:
        """
        Version simplifiée qui retourne seulement les résultats des agents.
        Compatible avec votre code existant.
        """
        logger.info("🚀 Démarrage du pipeline simplifié (backward compatible)")
        return self._run_agents(context, vector_store)