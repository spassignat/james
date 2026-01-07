# src/documentation/pipeline.py
import datetime
import json
from typing import Dict, List, Any

from documentation.md__template_generator import TemplateDocumentationGenerator
from models.project_structure import ProjectStructure


class DocumentationPipeline:
    """Pipeline complet de génération de documentation avec templates"""

    def __init__(self,
                 template_dir: str = "templates",
                 output_dir: str = "documentation",
                 custom_variables: Dict[str, Any] = None):

        self.generator = TemplateDocumentationGenerator(template_dir, output_dir)
        self.custom_variables = custom_variables or {}
        self.generated_docs: Dict[str, Any] = {}

    def generate_from_structure(self, project_structure: ProjectStructure) -> Dict[str, Any]:
        """
        Génère la documentation à partir d'une ProjectStructure
        """
        print(f"📝 Génération de documentation pour: {project_structure.project_name}")

        # Charger les templates
        self.generator.load_templates()

        # Configurer les variables
        self.generator.set_project_structure(project_structure)
        self.generator.set_custom_variables(self.custom_variables)

        # Générer la documentation
        self.generated_docs = self.generator.generate_all()

        print(f"✅ Documentation générée: {len(self.generated_docs)} fichiers")
        print(f"📁 Emplacement: {self.generator.output_dir.absolute()}")

        return self.generated_docs

    def generate_specific_docs(self,
                               project_structure: ProjectStructure,
                               templates: List[str]) -> Dict[str, Any]:
        """
        Génère seulement certains documents spécifiques
        """
        # Configurer les variables
        self.generator.set_project_structure(project_structure)
        self.generator.set_custom_variables(self.custom_variables)

        # Générer les documents spécifiques
        return self.generator.generate_specific(templates)

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la génération"""
        return {
            "output_dir": str(self.generator.output_dir.absolute()),
            "templates_used": list(self.generator.templates.keys()),
            "documents_generated": list(self.generated_docs.keys()),
            "total_documents": len(self.generated_docs),
            "variables_count": len(self.generator.variables)
        }

    def save_generation_report(self, project_structure: ProjectStructure):
        """Sauvegarde un rapport de génération"""
        report = {
            "project": project_structure.project_name,
            "generated_at": datetime.datetime.now().isoformat(),
            "documents": list(self.generated_docs.keys()),
            "statistics": {
                "files": len(project_structure.files),
                "modules": len(project_structure.modules),
                "patterns": len(project_structure.patterns_identified),
                "dependencies": len(project_structure.dependencies),
            }
        }

        report_path = self.generator.output_dir / "generation-report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))