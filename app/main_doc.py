# src/documentation/rule_generator.py
import os
from datetime import datetime
from typing import Dict, Any
import logging

import markdown

logger = logging.getLogger(__name__)
# src/documentation/rule_generator.py
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)


class RuleGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('analysis', {}).get('output_directory', './documentation')
        self.templates_dir = config.get('analysis', {}).get('templates_directory', './templates')

        # Créer les répertoires nécessaires
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)

        # Initialiser les templates par défaut si nécessaire
        self._init_default_templates()

        logger.info(f"✅ RuleGenerator initialisé - Sortie: {self.output_dir}")

    def _init_default_templates(self):
        """Initialise les templates par défaut si non présents"""
        default_templates = {
            'main.md': self._get_default_main_template(),
            'architecture.md': self._get_default_architecture_template(),
            'patterns.md': self._get_default_patterns_template(),
            'rules.md': self._get_default_rules_template(),
            'summary.md': self._get_default_summary_template()
        }

        for filename, content in default_templates.items():
            template_path = Path(self.templates_dir) / filename
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Template créé: {filename}")

    def generate_rules_documentation(self,
                                     analysis_results: Dict[str, Any],
                                     format: str = "markdown",
                                     include_metadata: bool = True) -> Dict[str, str]:
        """Génère la documentation dans différents formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"coding_rules_{timestamp}"

        outputs = {}

        if format in ["markdown", "all"]:
            md_output = self._generate_markdown_documentation(analysis_results, base_filename)
            outputs['markdown'] = md_output

        if format in ["html", "all"]:
            html_output = self._generate_html_documentation(analysis_results, base_filename)
            outputs['html'] = html_output

        if format in ["json", "all"]:
            json_output = self._generate_json_documentation(analysis_results, base_filename)
            outputs['json'] = json_output

        if format in ["yaml", "all"]:
            yaml_output = self._generate_yaml_documentation(analysis_results, base_filename)
            outputs['yaml'] = yaml_output

        # Générer un rapport de synthèse
        summary = self._generate_summary_report(analysis_results, base_filename, outputs)
        outputs['summary'] = summary

        logger.info(f"📄 Documentation générée dans {len(outputs)} formats")
        return outputs

    def _generate_markdown_documentation(self, results: Dict[str, Any], base_filename: str) -> str:
        """Génère la documentation markdown complète"""
        output_file = Path(self.output_dir) / f"{base_filename}.md"

        # Essayer d'utiliser un template personnalisé
        template_path = Path(self.templates_dir) / "main.md"
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

            # Créer le contexte pour le template
            context = self._build_template_context(results)

            # Rendre le template avec Jinja2
            template = Template(template_content)
            content = template.render(**context)
        else:
            # Utiliser la génération par défaut
            content = self._build_markdown_content(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"📝 Documentation markdown générée: {output_file}")
        return str(output_file)

    def _build_markdown_content(self, results: Dict[str, Any]) -> str:
        """Construit le contenu markdown structuré"""

        # Métadonnées
        metadata = self._extract_metadata(results)

        # Table des matières
        toc = self._generate_table_of_contents(results)

        # Sections principales
        sections = [
            self._generate_title_section(metadata),
            self._generate_executive_summary(results),
            self._generate_project_overview(results),
            self._generate_architecture_section(results),
            self._generate_patterns_section(results),
            self._generate_rules_section(results),
            self._generate_conventions_section(results),
            self._generate_implementation_guide(results),
            self._generate_quality_metrics(results),
            self._generate_appendix(results)
        ]

        content = f"{toc}\n\n" + "\n\n".join(filter(None, sections))

        return content

    def _generate_html_documentation(self, results: Dict[str, Any], base_filename: str) -> str:
        """Convertit la documentation markdown en HTML"""
        md_file = Path(self.output_dir) / f"{base_filename}.md"

        # Lire le markdown
        with open(md_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # Convertir en HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'codehilite']
        )

        # Ajouter un style CSS
        html_full = self._wrap_html(html_content, results)

        # Sauvegarder
        html_file = Path(self.output_dir) / f"{base_filename}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_full)

        logger.info(f"🌐 Documentation HTML générée: {html_file}")
        return str(html_file)

    def _wrap_html(self, content: str, results: Dict[str, Any]) -> str:
        """Encapsule le contenu HTML avec un template"""
        css_style = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; padding-bottom: 5px; border-bottom: 1px solid #ecf0f1; }
            h3 { color: #7f8c8d; }
            .metadata { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .rule { background: #fff; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; }
            .rule-header { font-weight: bold; color: #2c3e50; }
            .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin-right: 5px; }
            .badge-architecture { background: #3498db; color: white; }
            .badge-pattern { background: #2ecc71; color: white; }
            .badge-rule { background: #e74c3c; color: white; }
            .code-block { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; 
                          font-family: 'Courier New', monospace; overflow-x: auto; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .toc { background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .toc ul { list-style-type: none; padding-left: 0; }
            .toc li { margin: 5px 0; }
            .summary { background: #e8f4fc; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
        """

        title = "Règles de Codage - Documentation"

        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {css_style}
        </head>
        <body>
            <h1>{title}</h1>
            <div class="metadata">
                <strong>Généré le:</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}<br>
                <strong>Projet analysé:</strong> {results.get('project_name', 'Non spécifié')}<br>
                <strong>Nombre de règles:</strong> {results.get('rules', {}).get('rules_count', 0)}
            </div>
            {content}
            <hr>
            <footer>
                <p><small>Document généré automatiquement par le système d'analyse rétrospective</small></p>
            </footer>
        </body>
        </html>
        """

        return html

    def _generate_json_documentation(self, results: Dict[str, Any], base_filename: str) -> str:
        """Génère la documentation au format JSON structuré"""
        structured_data = self._structure_results_for_json(results)

        output_file = Path(self.output_dir) / f"{base_filename}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 Documentation JSON générée: {output_file}")
        return str(output_file)

    def _generate_yaml_documentation(self, results: Dict[str, Any], base_filename: str) -> str:
        """Génère la documentation au format YAML"""
        structured_data = self._structure_results_for_yaml(results)

        output_file = Path(self.output_dir) / f"{base_filename}.yaml"

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(structured_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"📄 Documentation YAML générée: {output_file}")
        return str(output_file)

    def _structure_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Structure les résultats pour l'export JSON"""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'project_name': results.get('project_name', 'Unknown'),
                'analysis_type': 'retro_engineering'
            },
            'summary': self._extract_summary(results),
            'architecture': {
                'analysis': results.get('architecture', {}).get('content', ''),
                'patterns_identified': results.get('architecture', {}).get('patterns_identified', []),
                'recommendations': results.get('architecture', {}).get('recommendations', [])
            },
            'patterns': {
                'analysis': results.get('patterns', {}).get('content', ''),
                'patterns_list': results.get('patterns', {}).get('patterns_identified', []),
                'patterns_count': results.get('patterns', {}).get('patterns_count', 0)
            },
            'rules': {
                'analysis': results.get('rules', {}).get('content', ''),
                'rules_list': results.get('rules', {}).get('rules', []),
                'rules_count': results.get('rules', {}).get('rules_count', 0),
                'categories': results.get('rules', {}).get('categories', {})
            },
            'statistics': {
                'total_files': results.get('project_stats', {}).get('total_files', 0),
                'total_rules': results.get('rules', {}).get('rules_count', 0),
                'total_patterns': results.get('patterns', {}).get('patterns_count', 0)
            }
        }

    def _structure_results_for_yaml(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Structure les résultats pour l'export YAML"""
        # Même structure que JSON mais formaté pour YAML
        return self._structure_results_for_json(results)

    def _generate_summary_report(self, results: Dict[str, Any], base_filename: str,
                                 outputs: Dict[str, str]) -> str:
        """Génère un rapport de synthèse"""
        summary_content = f"""
        # RAPPORT DE SYNTHÈSE - Génération de Documentation
        
        ## 📋 Informations Générales
        - **Date de génération**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        - **Projet analysé**: {results.get('project_name', 'Non spécifié')}
        - **Répertoire de sortie**: {self.output_dir}
        
        ## 📊 Résultats de l'Analyse
        - **Règles générées**: {results.get('rules', {}).get('rules_count', 0)}
        - **Patterns identifiés**: {results.get('patterns', {}).get('patterns_count', 0)}
        - **Fichiers analysés**: {results.get('project_stats', {}).get('total_files', 0)}
        
        ## 📁 Fichiers Générés
        {self._format_generated_files(outputs)}
        
        ## 🔍 Détails des Formats
        
        ### Markdown
        - **Fichier**: {outputs.get('markdown', 'Non généré')}
        - **Utilisation**: Documentation principale, facile à lire et modifier
        
        ### HTML
        - **Fichier**: {outputs.get('html', 'Non généré')}
        - **Utilisation**: Visualisation web, partage facile
        
        ### JSON
        - **Fichier**: {outputs.get('json', 'Non généré')}
        - **Utilisation**: Intégration avec d'autres outils, traitement automatique
        
        ### YAML
        - **Fichier**: {outputs.get('yaml', 'Non généré')}
        - **Utilisation**: Configuration, intégration avec systèmes CI/CD
        
        ## 🎯 Prochaines Étapes Recommandées
        1. Examiner les règles générées
        2. Adapter les règles au contexte spécifique du projet
        3. Intégrer les règles dans le pipeline de développement
        4. Planifier des revues de code basées sur ces règles
        5. Mettre à jour la documentation régulièrement
        
        ## 📞 Support
        Pour toute question ou suggestion concernant cette documentation:
        - Consulter la documentation générée
        - Réviser les règles pour adaptation
        - Contacter l'équipe d'architecture
        """

        summary_file = Path(self.output_dir) / f"{base_filename}_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        logger.info(f"📋 Rapport de synthèse généré: {summary_file}")
        return str(summary_file)

    def _extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les métadonnées des résultats"""
        return {
            'generation_date': datetime.now().strftime('%d/%m/%Y à %H:%M:%S'),
            'project_name': results.get('project_name', 'Projet non nommé'),
            'total_files': results.get('project_stats', {}).get('total_files', 0),
            'total_rules': results.get('rules', {}).get('rules_count', 0),
            'total_patterns': results.get('patterns', {}).get('patterns_count', 0),
            'agents_executed': results.get('agents_executed', []),
            'analysis_duration': self._calculate_analysis_duration(results)
        }

    def _generate_table_of_contents(self, results: Dict[str, Any]) -> str:
        """Génère une table des matières détaillée"""
        toc = [
            "# 📋 Table des Matières",
            "",
            "## 🎯 Vue d'ensemble",
            "- [Résumé Exécutif](#résumé-exécutif)",
            "- [Aperçu du Projet](#aperçu-du-projet)",
            "",
            "## 🏗️ Architecture",
            "- [Analyse Architecturale](#analyse-architecturale)",
            "- [Patterns Architecturaux](#patterns-architecturaux)",
            "- [Recommandations](#recommandations-architecturales)",
            "",
            "## 🔍 Patterns",
            "- [Patterns de Conception](#patterns-de-conception)",
            "- [Conventions Identifiées](#conventions-identifiées)",
            "",
            "## 📝 Règles de Codage",
            f"- [Règles Générées ({results.get('rules', {}).get('rules_count', 0)})](#règles-de-codage)",
            "- [Catégories de Règles](#catégories-de-règles)",
            "",
            "## 🎯 Conventions",
            "- [Conventions Recommandées](#conventions-recommandées)",
            "- [Standards de Qualité](#standards-de-qualité)",
            "",
            "## 🛠️ Guide d'Implémentation",
            "- [Mise en Œuvre](#mise-en-œuvre)",
            "- [Vérification](#vérification)",
            "",
            "## 📊 Métriques",
            "- [Statistiques](#statistiques)",
            "- [Qualité](#qualité)",
            "",
            "## 📎 Annexes",
            "- [Glossaire](#glossaire)",
            "- [Références](#références)",
        ]

        return '\n'.join(toc)

    def _generate_title_section(self, metadata: Dict[str, Any]) -> str:
        """Génère la section titre"""
        return f"""# 📚 Règles de Codage - Analyse Rétrospective

*Document généré automatiquement*

**Date de génération**: {metadata['generation_date']}  
**Projet**: {metadata['project_name']}  
**Fichiers analysés**: {metadata['total_files']}  
**Règles générées**: {metadata['total_rules']}  
**Patterns identifiés**: {metadata['total_patterns']}

---

"""

    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Génère le résumé exécutif"""
        summary = results.get('summary', {})

        return f"""## 🎯 Résumé Exécutif

### 📊 En Bref
- **État de l'analyse**: {results.get('status', 'Complété')}
- **Agents exécutés**: {', '.join(results.get('agents_executed', []))}
- **Durée d'analyse**: {results.get('analysis_duration', 'Non calculée')}

### 🏆 Points Forts Identifiés
{self._format_list(summary.get('strengths', ['À déterminer']))}

### 🎯 Domaines d'Amélioration
{self._format_list(summary.get('improvement_areas', ['À déterminer']))}

### 🚀 Recommandations Clés
{self._format_list(summary.get('key_recommendations', ['À déterminer']))}

"""

    def _generate_project_overview(self, results: Dict[str, Any]) -> str:
        """Génère l'aperçu du projet"""
        project_stats = results.get('project_stats', {})

        return f"""## 📁 Aperçu du Projet

### 📊 Statistiques
- **Total fichiers**: {project_stats.get('total_files', 0)}
- **Extensions principales**: {', '.join(project_stats.get('extensions', []))}
- **Structure**: {project_stats.get('structure_type', 'Standard')}

### 🗂️ Organisation
{self._format_directory_structure(project_stats.get('directory_structure', {}))}

"""

    def _generate_architecture_section(self, results: Dict[str, Any]) -> str:
        """Génère la section architecture"""
        arch = results.get('architecture', {})

        return f"""## 🏗️ Analyse Architecturale

### 🏛️ Vue d'ensemble
{arch.get('content', 'Non disponible')}

### 🧩 Patterns Architecturaux
{self._format_patterns(arch.get('patterns_identified', []))}

### 💡 Recommandations Architecturales
{self._format_recommendations(arch.get('recommendations', []))}

"""

    def _generate_patterns_section(self, results: Dict[str, Any]) -> str:
        """Génère la section patterns"""
        patterns = results.get('patterns', {})

        return f"""## 🔍 Patterns de Conception

### 📝 Analyse
{patterns.get('content', 'Non disponible')}

### 🎯 Patterns Identifiés
**Total**: {patterns.get('patterns_count', 0)} patterns

{self._format_detailed_patterns(patterns.get('patterns_identified', []))}

### 📋 Conventions Identifiées
{self._format_list(patterns.get('conventions', []))}

"""

    def _generate_rules_section(self, results: Dict[str, Any]) -> str:
        """Génère la section règles"""
        rules_data = results.get('rules', {})
        rules_list = rules_data.get('rules', [])

        return f"""## 📝 Règles de Codage

### 📋 Vue d'ensemble
{rules_data.get('content', 'Non disponible')}

### 🏷️ Catégories de Règles
{self._format_rule_categories(rules_data.get('categories', {}))}

### 📜 Liste Complète des Règles
**Total**: {rules_data.get('rules_count', 0)} règles

{self._format_detailed_rules(rules_list)}

"""

    def _generate_conventions_section(self, results: Dict[str, Any]) -> str:
        """Génère la section conventions"""
        return f"""## 🎯 Conventions Recommandées

### 📝 Conventions Générales
{self._generate_conventions_summary(results)}

### ⭐ Standards de Qualité
{self._generate_quality_standards()}

"""

    def _generate_implementation_guide(self, results: Dict[str, Any]) -> str:
        """Génère le guide d'implémentation"""
        return f"""## 🛠️ Guide d'Implémentation

### 🚀 Mise en Œuvre
1. **Priorisation**: Commencer par les règles les plus critiques
2. **Intégration**: Ajouter progressivement aux outils existants
3. **Formation**: Former l'équipe aux nouvelles règles
4. **Surveillance**: Suivre l'adoption et l'impact

### 🔍 Vérification
- Utiliser des outils d'analyse statique
- Mettre en place des revues de code régulières
- Intégrer dans le pipeline CI/CD
- Mesurer la conformité régulièrement

"""

    def _generate_quality_metrics(self, results: Dict[str, Any]) -> str:
        """Génère les métriques de qualité"""
        return f"""## 📊 Métriques

### 📈 Statistiques
- **Couverture des règles**: À mesurer
- **Conformité actuelle**: À établir
- **Taux d'adoption**: À suivre

### 🎯 Qualité
- **Maintenabilité**: Amélioration prévue
- **Lisibilité**: Impact positif attendu
- **Consistance**: Augmentation prévue

"""

    def _generate_appendix(self, results: Dict[str, Any]) -> str:
        """Génère les annexes"""
        return f"""## 📎 Annexes

### 📖 Glossaire
- **Pattern**: Solution réutilisable à un problème courant
- **Convention**: Accord sur la façon de faire les choses
- **Règle**: Directive spécifique à suivre
- **Architecture**: Organisation structurelle du système

### 📚 Références
- Principes SOLID
- Design Patterns (GoF)
- Clean Code (Robert C. Martin)
- Architecture Patterns (Microsoft)

---

*Document généré automatiquement par le système d'analyse rétrospective*
"""

    def _format_list(self, items: List[str]) -> str:
        """Formate une liste pour markdown"""
        if not items:
            return "*Aucun élément*"
        return '\n'.join([f"- {item}" for item in items])

    def _format_patterns(self, patterns: List) -> str:
        """Formate les patterns"""
        if not patterns:
            return "*Aucun pattern identifié*"

        if isinstance(patterns[0], dict):
            return '\n'.join([f"- **{p.get('name', 'Pattern')}**: {p.get('description', '')}"
                              for p in patterns[:10]])
        else:
            return self._format_list(patterns[:10])

    def _format_detailed_patterns(self, patterns: List) -> str:
        """Formate les patterns de manière détaillée"""
        if not patterns:
            return "*Aucun pattern détaillé*"

        formatted = []
        for i, pattern in enumerate(patterns[:20], 1):
            if isinstance(pattern, dict):
                formatted.append(f"{i}. **{pattern.get('name', f'Pattern {i}')}**")
                formatted.append(f"   - Description: {pattern.get('description', 'Non spécifiée')}")
                if 'examples' in pattern:
                    formatted.append(f"   - Exemples: {pattern.get('examples', '')}")
            else:
                formatted.append(f"{i}. {pattern}")

        return '\n'.join(formatted)

    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Formate les recommandations"""
        if not recommendations:
            return "*Aucune recommandation spécifique*"

        formatted = []
        for i, rec in enumerate(recommendations[:10], 1):
            formatted.append(f"{i}. {rec}")

        return '\n'.join(formatted)

    def _format_rule_categories(self, categories: Dict[str, int]) -> str:
        """Formate les catégories de règles"""
        if not categories:
            return "*Aucune catégorie définie*"

        formatted = []
        total = sum(categories.values())
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            formatted.append(f"- **{category}**: {count} règles ({percentage:.1f}%)")

        return '\n'.join(formatted)

    def _format_detailed_rules(self, rules: List[Dict]) -> str:
        """Formate les règles de manière détaillée"""
        if not rules:
            return "*Aucune règle générée*"

        formatted = []
        for i, rule in enumerate(rules[:50], 1):
            title = rule.get('title', f'Règle {i}')
            section = rule.get('section', 'Général')
            description = rule.get('description', '')

            formatted.append(f"### {i}. {title}")
            formatted.append(f"**Catégorie**: {section}")
            formatted.append(f"**Description**: {description}")

            if rule.get('examples'):
                formatted.append("**Exemples**:")
                for example in rule['examples'][:2]:
                    formatted.append(f"  - {example}")

            formatted.append("")  # Ligne vide entre les règles

        return '\n'.join(formatted)

    def _format_directory_structure(self, structure: Dict) -> str:
        """Formate la structure des répertoires"""
        if not structure:
            return "*Structure non disponible*"

        formatted = ["```"]
        for path, info in list(structure.items())[:10]:  # Limiter à 10 répertoires
            indent = "  " * (path.count('/') if path != '/' else 0)
            formatted.append(f"{indent}{path}/ ({info.get('files_count', 0)} fichiers)")

        formatted.append("```")
        return '\n'.join(formatted)

    def _format_generated_files(self, outputs: Dict[str, str]) -> str:
        """Formate la liste des fichiers générés"""
        if not outputs:
            return "*Aucun fichier généré*"

        formatted = []
        for format_name, filepath in outputs.items():
            filename = Path(filepath).name if filepath else 'Non généré'
            formatted.append(f"- **{format_name.upper()}**: `{filename}`")

        return '\n'.join(formatted)

    def _calculate_analysis_duration(self, results: Dict[str, Any]) -> str:
        """Calcule la durée de l'analyse"""
        start = results.get('pipeline_start')
        end = results.get('pipeline_end')

        if start and end:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
                duration = end_dt - start_dt
                return str(duration)
            except:
                pass

        return "Non calculée"

    def _generate_conventions_summary(self, results: Dict[str, Any]) -> str:
        """Génère le résumé des conventions"""
        patterns = results.get('patterns', {})
        conventions = patterns.get('conventions', [])

        if conventions:
            return "Conventions identifiées dans le codebase:\n" + self._format_list(conventions)
        else:
            return """### Conventions Recommandées

#### 🎯 Nommage
- Utiliser des noms explicites et descriptifs
- Suivre les conventions du langage/framework
- Éviter les abréviations obscures

#### 🏗️ Structure
- Organiser les fichiers par fonctionnalité
- Séparer les responsabilités
- Garder les fichiers à une taille raisonnable

#### 📝 Documentation
- Documenter les APIs publiques
- Ajouter des commentaires pour le code complexe
- Maintenir les README à jour

#### 🧪 Tests
- Un test par fonctionnalité
- Tests indépendants et reproductibles
- Nommage descriptif des tests
"""

    def _generate_quality_standards(self) -> str:
        """Génère les standards de qualité"""
        return """### Standards de Qualité Recommandés

#### 🎯 Lisibilité
- Code auto-documenté
- Structure claire et logique
- Commentaires pour les décisions complexes

#### 🛡️ Robustesse
- Gestion d'erreurs appropriée
- Validation des entrées
- Tests de cas limites

#### 🔄 Maintenabilité
- Faible couplage
- Haute cohésion
- Documentation à jour

#### ⚡ Performance
- Algorithmes optimisés
- Gestion efficace de la mémoire
- Profilage régulier
"""

    def _build_template_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Construit le contexte pour les templates"""
        return {
            'metadata': self._extract_metadata(results),
            'results': results,
            'generation_date': datetime.now().strftime('%d/%m/%Y à %H:%M:%S'),
            'formatters': {
                'list': self._format_list,
                'patterns': self._format_patterns,
                'rules': self._format_detailed_rules,
                'recommendations': self._format_recommendations
            }
        }

    # Méthodes pour les templates par défaut
    def _get_default_main_template(self) -> str:
        return """# {{ metadata.project_name }} - Règles de Codage

*Généré le {{ generation_date }}*

## 📋 Table des Matières
{{ formatters.list(results.get('agents_executed', [])) }}

## 🏗️ Architecture
{{ results.architecture.content if results.architecture else 'Non disponible' }}

## 🔍 Patterns
{{ results.patterns.content if results.patterns else 'Non disponible' }}

## 📝 Règles
{{ results.rules.content if results.rules else 'Non disponible' }}

## 📊 Statistiques
- Fichiers analysés: {{ metadata.total_files }}
- Règles générées: {{ metadata.total_rules }}
- Patterns identifiés: {{ metadata.total_patterns }}
"""

    def _get_default_architecture_template(self) -> str:
        return """# Analyse Architecturale

## Vue d'ensemble
{{ content }}

## Patterns Identifiés
{{ formatters.patterns(patterns_identified) }}

## Recommandations
{{ formatters.recommendations(recommendations) }}
"""

    def _get_default_patterns_template(self) -> str:
        return """# Analyse des Patterns

## Patterns de Conception
{{ content }}

## Liste des Patterns
{% for pattern in patterns_identified %}
### {{ pattern.name }}
{{ pattern.description }}
{% endfor %}
"""

    def _get_default_rules_template(self) -> str:
        return """# Règles de Codage

## Vue d'ensemble
{{ content }}

## Liste des Règles
{{ formatters.rules(rules) }}
"""

    def _get_default_summary_template(self) -> str:
        return """# Résumé de l'Analyse

## Métriques
- Total règles: {{ rules_count }}
- Total patterns: {{ patterns_count }}
- Fichiers analysés: {{ total_files }}

## Recommandations Principales
{{ formatters.recommendations(key_recommendations) }}
"""