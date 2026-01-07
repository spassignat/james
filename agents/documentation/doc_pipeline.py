import datetime
import json
from typing import Dict

from models.project_structure import ProjectStructure


class DocumentationPipeline:
    """Pipeline qui intègre les agents et génère la documentation"""

    def __init__(self):
        self.agents_results = []
        self.project_structure = None

    def add_agent_result(self, agent_name: str, result: Dict):
        """Ajoute un résultat d'agent"""
        self.agents_results.append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "data": result
        })

    def merge_results(self) -> ProjectStructure:
        """Fusionne les résultats de tous les agents"""
        if not self.agents_results:
            raise ValueError("Aucun résultat d'agent à fusionner")

        # Créer la structure de base
        structure = ProjectStructure(
            name=self._extract_project_name(),
            description=self._extract_description(),
            agents_used=[r["agent"] for r in self.agents_results],
            timestamp=datetime.now().isoformat()
        )

        # Fusionner les fichiers de tous les agents
        for result in self.agents_results:
            if "files" in result["data"]:
                for path, file_data in result["data"]["files"].items():
                    if path not in structure.files:
                        structure.files[path] = FileInfo(path=path)

                    # Mettre à jour les informations
                    file_info = structure.files[path]
                    if "content" in file_data and file_data["content"]:
                        file_info.content = file_data["content"]
                    if "description" in file_data:
                        file_info.description = file_data["description"]
                    if "language" in file_data:
                        file_info.language = file_data["language"]
                    if "functions" in file_data:
                        file_info.functions.extend(file_data["functions"])
                    if "classes" in file_data:
                        file_info.classes.extend(file_data["classes"])

        # Extraire l'architecture
        for result in self.agents_results:
            if "architecture" in result["data"]:
                structure.architecture.update(result["data"]["architecture"])

        # Extraire les dépendances
        for result in self.agents_results:
            if "dependencies" in result["data"]:
                structure.dependencies.extend(result["data"]["dependencies"])

        self.project_structure = structure
        return structure

    def generate_documentation(self) -> Dict[str, str]:
        """Génère toute la documentation"""
        if not self.project_structure:
            self.merge_results()

        generator = MarkdownDocumentationGenerator(self.project_structure)
        return generator.generate_all()

    def save_project_structure(self, path: str = "project_structure.json"):
        """Sauvegarde la structure du projet"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.project_structure.to_dict(), f, indent=2, ensure_ascii=False)

    def load_project_structure(self, path: str = "project_structure.json"):
        """Charge la structure du projet"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.project_structure = ProjectStructure.from_dict(data)

    def _extract_project_name(self) -> str:
        """Extrait le nom du projet des résultats"""
        for result in self.agents_results:
            if "name" in result["data"]:
                return result["data"]["name"]
        return "Unnamed Project"

    def _extract_description(self) -> str:
        """Extrait la description des résultats"""
        descriptions = []
        for result in self.agents_results:
            if "description" in result["data"]:
                descriptions.append(result["data"]["description"])
        return " ".join(descriptions) if descriptions else ""