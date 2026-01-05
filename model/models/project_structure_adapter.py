# src/documentation/structure_adapter.py
from typing import Dict, List, Any
from ..models.project_structure import ProjectStructure

class ProjectStructureAdapter:
    """Adapte votre ProjectStructure existante pour la génération de docs"""

    @staticmethod
    def to_extended_structure(project: ProjectStructure) -> Dict[str, Any]:
        """
        Convertit votre ProjectStructure en format étendu pour la documentation
        """
        # Structure de base
        extended = {
            "name": project.project_name,
            "description": project.architecture_overview or f"Projet {project.project_name}",
            "version": "1.0.0",
            "files": {},
            "architecture": {
                "overview": project.architecture_overview,
                "patterns": project.patterns_identified,
                "components": {}
            },
            "dependencies": project.dependencies,
            "agents_used": ["ArchitectureAgent", "PatternAgent", "DependencyAgent"],
            "timestamp": project.last_analysis_at,
            "metadata": project.extra_metadata
        }

        # Convertir les composants
        for comp_type, comp_names in project.components.items():
            extended["architecture"]["components"][comp_type] = [
                {"name": name, "description": f"Composant {comp_type}: {name}"}
                for name in comp_names
            ]

        # Convertir les fichiers (structure simplifiée)
        for file_path in project.files:
            extended["files"][file_path] = {
                "path": file_path,
                "language": ProjectStructureAdapter._detect_language(file_path),
                "description": ProjectStructureAdapter._generate_file_description(file_path, project),
                "dependencies": [],
                "functions": [],
                "classes": [],
                "tags": []
            }

        return extended

    @staticmethod
    def _detect_language(file_path: str) -> str:
        """Détecte le langage basé sur l'extension"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.toml': 'toml',
            '.sql': 'sql',
            '.sh': 'bash'
        }

        for ext, lang in extensions.items():
            if file_path.endswith(ext):
                return lang
        return 'unknown'

    @staticmethod
    def _generate_file_description(file_path: str, project: ProjectStructure) -> str:
        """Génère une description basique pour un fichier"""
        filename = file_path.split('/')[-1]

        # Essayer de deviner le type basé sur le nom
        if any(keyword in filename.lower() for keyword in ['test', 'spec']):
            return f"Tests pour {file_path}"
        elif any(keyword in filename.lower() for keyword in ['config', 'settings']):
            return f"Configuration {project.project_name}"
        elif any(keyword in filename.lower() for keyword in ['readme', 'license']):
            return f"Documentation {project.project_name}"
        elif any(keyword in filename.lower() for keyword in ['main', 'app', 'index']):
            return f"Point d'entrée {project.project_name}"

        return f"Fichier {filename}"