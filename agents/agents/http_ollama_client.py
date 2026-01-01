# src/utils/ollama_client.py
import json
import logging
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)


class HTTPOllamaClient:
    """Client HTTP pour l'API Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')

    def generate_text(self,
                      model: str,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.1,
                      max_tokens: int = 8192,
                      timeout: int = 60) -> str:
        """Génère du texte via l'API Ollama"""
        try:
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            return data.get('response', '')

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur requête Ollama: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur parsing JSON Ollama: {e}")
            return ""

    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Chat avec l'API Ollama"""
        try:
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }

            if options:
                payload["options"] = options

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur chat Ollama: {e}")
            return {"message": {"content": ""}}

    def get_models(self) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles disponibles"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('models', [])

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur récupération modèles Ollama: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Télécharge un modèle"""
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}

            response = requests.post(url, json=payload, timeout=300)  # 5min timeout
            response.raise_for_status()

            logger.info(f"✅ Modèle {model_name} téléchargé")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur téléchargement modèle {model_name}: {e}")
            return False
