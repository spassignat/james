# src/project_analyzer.py
import logging
from pathlib import Path
from typing import List, Dict, Any

from models.code_chunk import CodeChunk
from models.project_structure import ProjectStructure
from parsers.utils.Util import infer_language_from_path, infer_category_from_type

logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    """
    Analyse la structure d'un projet et produit :
    - ProjectStructure : arborescence, patterns, statistiques
    - Liste de CodeChunk pour chaque fichier analysé
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_path = Path(config.get("project_root", "."))
        self.chunks: List[CodeChunk] = []

    def analyze_project_structure(self) -> ProjectStructure:
        """
        Analyse la structure du projet et retourne un objet ProjectStructure typé
        """
        logger.info(f"🔍 Analyse de la structure du projet: {self.root_path}")

        files = self._collect_files(self.root_path)
        patterns = self._infer_patterns(files)

        project_structure = ProjectStructure(
            name=self.root_path.name,
            root_path=str(self.root_path),
            files=files,
            patterns=patterns
        )

        # Extraire les chunks pour chaque fichier
        for file_path in files:
            file_chunks = self._extract_chunks(file_path)
            self.chunks.extend(file_chunks)

        logger.info(f"✅ Structure projet analysée : {len(files)} fichiers, {len(self.chunks)} chunks extraits")
        return project_structure

    def _collect_files(self, path: Path) -> List[str]:
        """
        Récupère tous les fichiers pertinents du projet
        """
        exts = self.config.get("file_extensions", [".py", ".js", ".ts", ".java", ".vue"])
        files = [str(f) for f in path.rglob("*") if f.suffix in exts and f.is_file()]
        logger.debug(f"📂 Fichiers collectés : {len(files)}")
        return files

    def _infer_patterns(self, files: List[str]) -> Dict[str, Any]:
        """
        Déduit des patterns dans le projet (par exemple structure MVC, services, composants)
        """
        patterns = {}
        for file_path in files:
            fname = Path(file_path).name.lower()
            if "controller" in fname:
                patterns[file_path] = "controller"
            elif "service" in fname:
                patterns[file_path] = "service"
            elif "repository" in fname:
                patterns[file_path] = "repository"
            elif "component" in fname:
                patterns[file_path] = "component"
            else:
                patterns[file_path] = "business_logic"
        logger.debug(f"🧩 Patterns déduits : {len(patterns)}")
        return patterns

    def _extract_chunks(self, file_path: str) -> List[CodeChunk]:
        """
        Transforme le fichier en chunks de code typés
        """
        chunks: List[CodeChunk] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Pour simplifier, on découpe par lignes ou blocs (à adapter)
            raw_chunks = content.split("\n\n")
            for i, block in enumerate(raw_chunks):
                chunk = CodeChunk(
                    content=block,
                    file_path=file_path,
                    filename=Path(file_path).name,
                    language=infer_language_from_path(file_path),
                    category=infer_category_from_type("", file_path),
                    chunk_type="code_block"
                )
                chunks.append(chunk)

        except Exception as e:
            logger.warning(f"⚠️ Impossible d'extraire chunks de {file_path}: {e}")

        return chunks

    def get_chunks(self) -> List[CodeChunk]:
        """Retourne tous les chunks extraits"""
        return self.chunks
