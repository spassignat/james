import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from vector.vector_store import VectorStore

logger = logging.getLogger(__name__)


class PatternAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        # CORRECTION: Appel correct au parent
        super().__init__(config, 'pattern_agent')

    def analyze(self, context: Dict[str, Any], vector_store: VectorStore) -> Dict[str, Any]:
        """Analyse les patterns spécifiques par type de fichier"""
        logger.info(f"🔍 Début analyse patterns par {self.agent_name}")

        file_patterns = context.get('file_patterns', {})
        chunks = context.get('chunks', [])

        prompt = self._build_pattern_prompt(file_patterns, chunks)

        system_prompt = """Vous êtes un expert en patterns de conception et en analyse de code.
Vous devez identifier les patterns de conception, les conventions de nommage et les bonnes pratiques
dans un codebase. Soyez précis et technique."""

        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.agent_config.get('temperature', 0.2)
        )

        patterns_identified = self._extract_patterns_from_response(response)

        return {
            'type': 'pattern_analysis',
            'agent': self.agent_name,
            'timestamp': self._get_timestamp(),
            'content': response,
            'patterns_identified': patterns_identified,
            'patterns_count': len(patterns_identified),
            'conventions': self._extract_conventions(response)
        }

    def _build_pattern_prompt(self, patterns: Dict, chunks: List) -> str:
        """Construit le prompt pour l'analyse de patterns"""

        # Échantillonner des chunks représentatifs
        sampled_chunks = self._sample_code_patterns(chunks)

        prompt = f"""
# ANALYSE DE PATTERNS DE CONCEPTION

## CONTEXTE
Analyse des patterns de conception et conventions dans un projet.

## 1. PATTERNS DE FICHIERS OBSERVÉS
{self._format_patterns_for_analysis(patterns)}

## 2. EXTRAITS DE CODE POUR ANALYSE
{sampled_chunks}

## TÂCHE D'ANALYSE

Identifiez et analysez les patterns de conception, conventions de nommage et bonnes pratiques
dans ce codebase.

### Points à analyser:

1. **Design Patterns**
   - Patterns de création (Factory, Singleton, Builder...)
   - Patterns structurels (Adapter, Decorator, Composite...)
   - Patterns comportementaux (Observer, Strategy, Command...)

2. **Patterns architecturaux**
   - Patterns d'organisation des couches
   - Patterns de communication entre composants
   - Patterns de persistance des données

3. **Conventions de nommage**
   - Conventions pour les classes, méthodes, variables
   - Conventions pour les fichiers et répertoires
   - Conventions spécifiques au framework/langage

4. **Patterns de code**
   - Patterns de gestion d'erreurs
   - Patterns de validation
   - Patterns de logique métier récurrents

5. **Bonnes pratiques identifiées**
   - Principes SOLID respectés/violés
   - DRY (Don't Repeat Yourself)
   - KISS (Keep It Simple, Stupid)
   - YAGNI (You Ain't Gonna Need It)

6. **Recommandations**
   - Patterns à renforcer
   - Anti-patterns à corriger
   - Améliorations possibles

### Format de réponse:
Fournissez une analyse structurée en Markdown.
"""
        return prompt

    def _format_patterns_for_analysis(self, patterns: Dict) -> str:
        """Formate les patterns pour l'analyse"""
        if not patterns:
            return "Aucun pattern pré-identifié."

        formatted = []
        for pattern_type, data in patterns.items():
            if isinstance(data, dict):
                formatted.append(f"### {pattern_type}")
                for key, value in data.items():
                    formatted.append(f"- **{key}**: {value}")
            elif isinstance(data, list):
                formatted.append(f"### {pattern_type} ({len(data)} éléments)")
                for item in data[:5]:
                    formatted.append(f"- {item}")
            else:
                formatted.append(f"### {pattern_type}: {data}")

        return '\n'.join(formatted)

    def _sample_code_patterns(self, chunks: List, max_samples: int = 15) -> str:
        """Échantillonne des chunks pour l'analyse de patterns"""
        if not chunks:
            return "Aucun extrait de code disponible pour l'analyse de patterns."

        # Filtrer les chunks intéressants pour l'analyse de patterns
        interesting_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                # Prioriser les chunks avec des patterns potentiels
                if any(keyword in content.lower() for keyword in
                       ['class', 'function', 'interface', 'extends', 'implements',
                        'factory', 'service', 'repository', 'controller']):
                    interesting_chunks.append(chunk)

        if not interesting_chunks:
            interesting_chunks = chunks[:max_samples]
        else:
            interesting_chunks = interesting_chunks[:max_samples]

        formatted = []
        for i, chunk in enumerate(interesting_chunks):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')[:250]
                metadata = chunk.get('metadata', {})
                filename = metadata.get('filename', 'Inconnu')
                language = metadata.get('language', 'code')

                formatted.append(f"#### Extrait {i + 1} - {filename} ({language})")
                formatted.append(f"```{language}\n{content}...\n```")

        return '\n\n'.join(formatted) if formatted else "Aucun extrait de code significatif pour l'analyse de patterns."

    def _extract_patterns_from_response(self, response: str) -> List[Dict[str, str]]:
        """Extrait les patterns de la réponse LLM"""
        patterns = []
        lines = response.split('\n')

        current_pattern = None
        for line in lines:
            line_stripped = line.strip()

            # Détecter un nouveau pattern
            if line_stripped.startswith('**') and line_stripped.endswith('**'):
                if current_pattern:
                    patterns.append(current_pattern)
                pattern_name = line_stripped.strip('*').strip()
                current_pattern = {'name': pattern_name, 'description': ''}
            elif current_pattern and line_stripped and not line_stripped.startswith('#'):
                if 'description' in current_pattern:
                    current_pattern['description'] += ' ' + line_stripped
                else:
                    current_pattern['description'] = line_stripped

        if current_pattern:
            patterns.append(current_pattern)

        return patterns if patterns else [
            {'name': 'Patterns non explicitement identifiés',
             'description': 'Voir le contenu complet pour l\'analyse détaillée.'}
        ]

    def _extract_conventions(self, response: str) -> List[str]:
        """Extrait les conventions identifiées"""
        conventions = []
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            convention_keywords = ['convention', 'naming', 'standard', 'best practice',
                                   'recommend', 'should', 'must', 'always', 'never']
            if any(keyword in line_lower for keyword in convention_keywords):
                conventions.append(line.strip())

        return conventions[:10]  # Limiter à 10 conventions
