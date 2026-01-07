# src/documentation/markdown_generator.py
from pathlib import Path
from typing import Dict

from models.project_structure import ProjectStructure
from models.project_structure_adapter import ProjectStructureAdapter


class MarkdownDocumentationGenerator:
    """Génère une documentation Markdown à partir de votre ProjectStructure"""

    def __init__(self, project_structure: ProjectStructure, output_dir: str = "docs"):
        self.project = project_structure
        self.extended = ProjectStructureAdapter.to_extended_structure(project_structure)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_all(self) -> Dict[str, str]:
        """Génère toute la documentation"""
        docs = {}

        # Documentation principale
        docs["README"] = self.generate_readme()
        docs["ARCHITECTURE"] = self.generate_architecture()
        docs["API"] = self.generate_api_reference()
        docs["SETUP"] = self.generate_setup_guide()
        docs["DEVELOPMENT"] = self.generate_development_guide()

        # Sauvegarder
        self._save_docs(docs)

        return docs

    def generate_readme(self) -> str:
        """Génère le README.md principal"""
        project = self.project

        return f"""# {project.project_name}

{project.architecture_overview}

## 📊 Vue Rapide

- **Modules**: {len(project.modules)}
- **Fichiers**: {len(project.files)}
- **Patterns**: {len(project.patterns_identified)}
- **Dépendances**: {len(project.dependencies)}

## 🏗️ Architecture

{self._format_architecture_summary()}

## 🚀 Démarrage Rapide

```bash
# Cloner et installer
git clone <repository-url>
cd {project.project_name.lower().replace(' ', '-')}

# Installer les dépendances
{self._format_install_commands()}

# Lancer
{self._format_start_commands()}