from dataclasses import dataclass
from datetime import datetime
# src/models/project_structure.py
from typing import List, Dict, Any, Optional


@dataclass
class ProjectStructure:
    project_name: str
    modules: List[str]
    files: List[str]
    patterns_identified: List[str]
    architecture_overview: str
    components: Dict[str, Any]
    dependencies: List[str]
    created_at: str
    last_analysis_at: str
    extra_metadata: Dict[str, Any]
    """
    Représente la structure complète d'un projet pour l'analyse et la génération de documentation.
    Utilisé par les agents et le RuleGenerator.
    """

    def __init__(
            self,
            project_name: str,
            modules: Optional[List[str]] = None,
            files: Optional[List[str]] = None,
            patterns_identified: Optional[List[str]] = None,
            architecture_overview: str = "",
            components: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
            created_at: Optional[str] = None,
            last_analysis_at: Optional[str] = None,
            extra_metadata: Optional[Dict[str, Any]] = None
    ):
        # Nom du projet
        self.project_name: str = project_name

        # Liste des modules / packages détectés
        self.modules: List[str] = modules or []

        # Liste de tous les fichiers analysés
        self.files: List[str] = files or []

        # Patterns et design patterns identifiés
        self.patterns_identified: List[str] = patterns_identified or []

        # Vue d'ensemble de l'architecture (texte généré par ArchitectureAgent)
        self.architecture_overview: str = architecture_overview

        # Composants principaux et leurs responsabilités
        # Exemple: {"service": ["UserService", "AuthService"], "controller": ["UserController"]}
        self.components: Dict[str, Any] = components or {}

        # Dépendances externes / frameworks
        self.dependencies: List[str] = dependencies or []

        # Dates pour suivi et versioning
        self.created_at: str = created_at or datetime.now().isoformat()
        self.last_analysis_at: str = last_analysis_at or datetime.now().isoformat()

        # Métadonnées supplémentaires flexibles
        self.extra_metadata: Dict[str, Any] = extra_metadata or {}

    def add_module(self, module_name: str):
        if module_name not in self.modules:
            self.modules.append(module_name)

    def add_file(self, file_path: str):
        if file_path not in self.files:
            self.files.append(file_path)

    def add_pattern(self, pattern: str):
        if pattern not in self.patterns_identified:
            self.patterns_identified.append(pattern)

    def add_component(self, component_type: str, component_name: str):
        if component_type not in self.components:
            self.components[component_type] = []
        if component_name not in self.components[component_type]:
            self.components[component_type].append(component_name)

    def add_dependency(self, dependency_name: str):
        if dependency_name not in self.dependencies:
            self.dependencies.append(dependency_name)

    def update_last_analysis(self):
        self.last_analysis_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation complète pour JSON / sauvegarde / export"""
        return {
            "project_name": self.project_name,
            "modules": self.modules,
            "files": self.files,
            "patterns_identified": self.patterns_identified,
            "architecture_overview": self.architecture_overview,
            "components": self.components,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "last_analysis_at": self.last_analysis_at,
            "extra_metadata": self.extra_metadata
        }
