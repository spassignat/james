import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RulesAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        # CORRECTION: Appel correct au parent
        super().__init__(config, 'rules_agent')

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère les règles de codage finales"""
        logger.info(f"📋 Début génération de règles par {self.agent_name}")

        architecture_analysis = context.get('architecture_analysis', {})
        pattern_analysis = context.get('pattern_analysis', {})

        prompt = self._build_rules_prompt(architecture_analysis, pattern_analysis)

        system_prompt = """Vous êtes un expert en conventions de codage et standards de développement.
Vous devez créer un ensemble de règles de codage basées sur l'analyse d'un projet existant.
Les règles doivent être pratiques, applicables et spécifiques au contexte du projet."""

        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.agent_config.get('temperature', 0.1)
        )

        rules = self._extract_rules(response)

        return {
            'type': 'coding_rules',
            'agent': self.agent_name,
            'timestamp': self._get_timestamp(),
            'content': response,
            'rules': rules,
            'rules_count': len(rules),
            'categories': self._categorize_rules(rules)
        }

    def _build_rules_prompt(self, architecture: Dict, patterns: Dict) -> str:
        """Construit le prompt pour la génération de règles"""

        prompt = f"""
# GÉNÉRATION DE RÈGLES DE CODAGE

## CONTEXTE
Création d'un guide de règles de codage basé sur l'analyse d'un projet existant.

## 1. ANALYSE D'ARCHITECTURE
{self._format_analysis_for_rules(architecture)}

## 2. ANALYSE DE PATTERNS
{self._format_analysis_for_rules(patterns)}

## TÂCHE: CRÉER UN GUIDE DE RÈGLES DE CODAGE

Basé sur les analyses ci-dessus, créez un guide complet de règles de codage pour ce projet.

### Structure du guide:

1. **Règles d'architecture**
   - Organisation des modules/paquets
   - Séparation des responsabilités
   - Communication entre composants

2. **Règles de conception**
   - Utilisation des design patterns
   - Principes SOLID à appliquer
   - Contrats d'interface

3. **Conventions de code**
   - Conventions de nommage
   - Formatage du code
   - Structure des fichiers

4. **Règles spécifiques au langage**
   - Bonnes pratiques Java/JavaScript
   - Utilisation des frameworks
   - Gestion des dépendances

5. **Standards de qualité**
   - Gestion des erreurs
   - Logging
   - Tests
   - Documentation

6. **Règles de sécurité**
   - Validation des entrées
   - Gestion des authentifications
   - Protection des données

### Format des règles:
Chaque règle doit suivre ce format:
- **Titre clair et concis**
- **Description**: Explication de la règle
- **Exemple correct**: Code montrant comment appliquer la règle
- **Exemple incorrect**: Code montrant ce qu'il faut éviter
- **Justification**: Pourquoi cette règle est importante

### Exigences:
- Les règles doivent être concrètes et applicables
- Inclure des exemples de code pertinents
- Adapter les règles au contexte spécifique du projet
- Prioriser les règles les plus importantes
"""
        return prompt

    def _format_analysis_for_rules(self, analysis: Dict) -> str:
        """Formate une analyse pour la génération de règles"""
        if not analysis:
            return "Aucune analyse disponible."

        content = analysis.get('content', '')
        # Limiter la longueur
        if len(content) > 1000:
            content = content[:1000] + "...\n[Contenu tronqué]"

        summary = analysis.get('summary', '')
        patterns = analysis.get('patterns_identified', [])
        recommendations = analysis.get('recommendations', [])

        formatted = []
        if summary:
            formatted.append(f"**Résumé**: {summary}")

        if patterns:
            formatted.append(f"**Patterns identifiés**: {', '.join(patterns[:5])}")

        if recommendations:
            formatted.append("**Recommandations clés**:")
            for rec in recommendations[:3]:
                formatted.append(f"- {rec}")

        return '\n'.join(formatted)

    def _extract_rules(self, response: str) -> List[Dict[str, str]]:
        """Extrait les règles structurées de la réponse"""
        rules = []
        lines = response.split('\n')

        current_rule = None
        current_section = None

        for line in lines:
            line_stripped = line.strip()

            # Détecter les sections principales
            if line_stripped.startswith('## '):
                current_section = line_stripped[3:].strip()
                continue

            # Détecter une nouvelle règle
            if line_stripped.startswith('### ') or line_stripped.startswith('**'):
                if current_rule and 'description' in current_rule:
                    rules.append(current_rule)

                rule_title = line_stripped.strip('#*').strip()
                current_rule = {
                    'title': rule_title,
                    'section': current_section or 'Général',
                    'description': '',
                    'examples': []
                }

            # Ajouter du contenu à la règle courante
            elif current_rule:
                if line_stripped.startswith('- **Description**:'):
                    current_rule['description'] = line_stripped[17:].strip()
                elif line_stripped.startswith('- **Exemple'):
                    current_rule['examples'].append(line_stripped[3:].strip())
                elif line_stripped and not line_stripped.startswith('#') and len(line_stripped) > 10:
                    # Si pas de balise spécifique, ajouter à la description
                    if not current_rule['description']:
                        current_rule['description'] = line_stripped
                    else:
                        current_rule['description'] += ' ' + line_stripped

        # Ajouter la dernière règle
        if current_rule and 'description' in current_rule:
            rules.append(current_rule)

        # Si pas de règles structurées, créer des règles basiques
        if not rules:
            sections = response.split('## ')
            for section in sections[1:]:  # Ignorer le premier élément (vide)
                lines_section = section.split('\n')
                section_title = lines_section[0].strip()
                for line in lines_section[1:]:
                    line_stripped = line.strip()
                    if line_stripped.startswith('-') or line_stripped.startswith('*'):
                        rule_text = line_stripped[2:].strip()
                        if len(rule_text) > 20:  # Filtrer les lignes trop courtes
                            rules.append({
                                'title': f"Règle dans {section_title}",
                                'section': section_title,
                                'description': rule_text,
                                'examples': []
                            })

        return rules[:50]  # Limiter à 50 règles maximum

    def _categorize_rules(self, rules: List[Dict]) -> Dict[str, int]:
        """Catégorise les règles par section"""
        categories = {}
        for rule in rules:
            section = rule.get('section', 'Non catégorisé')
            categories[section] = categories.get(section, 0) + 1

        return categories
