# src/config_loader.py
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.info(f"Configuration chargée depuis: {self.config_path}")
            return config

        except FileNotFoundError:
            logger.error(f"Fichier de configuration non trouvé: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Erreur de parsing YAML: {e}")
            raise

    def get_project_config(self) -> Dict[str, Any]:
        return self.config.get('project', {})

    def get_vectorization_config(self) -> Dict[str, Any]:
        return self.config.get('vectorization', {})

    def get_ollama_config(self) -> Dict[str, Any]:
        return self.config.get('ollama', {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})