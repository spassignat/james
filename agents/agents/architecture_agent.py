import logging
from typing import Dict, Any, List
from pathlib import Path

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ArchitectureAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, 'architecture_agent')

        # Chargement du prompt depuis fichier
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Charge le prompt depuis prompt/analyse.md"""
        prompt_path = Path("prompt/analyse.md")

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"❌ Fichier de prompt introuvable : {prompt_path}"
            )

        return prompt_path.read_text(encoding="utf-8")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🏗️  Début analyse architecture par {self.agent_name}")

        project_structure = context.get('project_structure', {})
        chunks = context.get('chunks', [])
        file_patterns = context.get('file_patterns', {})

        prompt = self._build_prompt_from_template(
            structure=project_structure,
            chunks=chunks,
            file_patterns=file_patterns
        )

        system_prompt = (
            "Vous êtes un architecte logiciel expert spécialisé "
            "dans l’analyse de code et d’architecture applicative."
        )

        response = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt
        )

        return {
            'type': 'architecture_analysis',
            'agent': self.agent_name,
            'timestamp': self._get_timestamp(),
            'content': response,
            'summary': self._extract_summary(response),
            'recommendations': self._extract_recommendations(response),
            'patterns_identified': self._extract_patterns(response)
        }

    # ------------------------------------------------------------------
    # Prompt handling
    # ------------------------------------------------------------------

    def _build_prompt_from_template(
            self,
            structure: Dict,
            chunks: List,
            file_patterns: Dict
    ) -> str:
        """Injecte les données dans le prompt markdown"""

        return self.prompt_template \
            .replace("{{PROJECT_STRUCTURE}}", self._format_structure(structure)) \
            .replace("{{FILE_PATTERNS}}", self._format_file_patterns(file_patterns)) \
            .replace("{{CODE_CHUNKS}}", self._sample_chunks(chunks))

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_structure(self, structure: Dict) -> str:
        if not structure:
            return "Aucune structure fournie."

        lines = []
        for key, value in structure.items():
            if isinstance(value, list):
                lines.append(f"- **{key}** ({len(value)} éléments)")
                for v in value[:5]:
                    lines.append(f"  - {v}")
            elif isinstance(value, dict):
                lines.append(f"- **{key}** ({len(value)} sous-éléments)")
            else:
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def _sample_chunks(self, chunks: List, max_samples: int = 10) -> str:
        if not chunks:
            return "Aucun extrait de code."

        selected = chunks[:max_samples]

        formatted = []
        for i, chunk in enumerate(selected):
            if isinstance(chunk, dict):
                content = chunk.get("content", "")[:400]
                filename = chunk.get("metadata", {}).get("filename", "inconnu")
                formatted.append(
                    f"### Extrait {i+1} – {filename}\n```{content}```"
                )
            else:
                formatted.append(f"### Extrait {i+1}\n```{str(chunk)[:400]}```")

        return "\n\n".join(formatted)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_summary(self, response: str) -> str:
        for line in response.splitlines():
            if line.strip() and not line.startswith("#"):
                return line.strip()
        return "Résumé non détecté."

    def _extract_recommendations(self, response: str) -> List[str]:
        recs = []
        for line in response.splitlines():
            if line.strip().startswith("-"):
                recs.append(line.strip()[2:])
        return recs or ["Aucune recommandation détectée."]

    def _extract_patterns(self, response: str) -> List[str]:
        keywords = [
            "mvc", "hexagonal", "clean", "repository",
            "service", "factory", "ddd", "cqrs"
        ]
        return [
            line.strip()
            for line in response.splitlines()
            if any(k in line.lower() for k in keywords)
        ][:10]
