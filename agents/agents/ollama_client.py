# src/utils/ollama_client.py
import logging
from enum import Enum
from typing import Dict, Any, Optional, List

import requests

logger = logging.getLogger(__name__)

import os

os.environ['HF_HUB_OFFLINE'] = '1'  # Mode hors ligne
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


class OllamaAPIType(Enum):
    LEGACY_GENERATE = "api/generate"
    OPENAI_COMPATIBLE = "v1/chat/completions"
    LEGACY_CHAT = "api/chat"
    UNKNOWN = "unknown"


class AdaptiveOllamaClient:
    """Client Ollama qui détecte automatiquement l'API disponible"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.api_type = None
        self._discover_api()

    def _discover_api(self):
        """Découvre l'API Ollama disponible"""
        test_endpoints = [
            (OllamaAPIType.LEGACY_CHAT, "/api/chat"),
            (OllamaAPIType.LEGACY_GENERATE, "/api/generate"),
            (OllamaAPIType.OPENAI_COMPATIBLE, "/v1/chat/completions"),
        ]

        for api_type, endpoint in test_endpoints:
            url = f"{self.base_url}{endpoint}"
            try:
                # Tester avec une petite requête
                test_payload = {"model": "llama2", "prompt": "test"}
                response = requests.post(url, json=test_payload, timeout=3)

                if response.status_code in [200, 400, 404]:  # 404 si mauvais modèle, mais endpoint existe
                    self.api_type = api_type
                    logger.debug(url)
                    logger.info(f"✅ API Ollama détectée: {api_type.value}")
                    return

            except requests.exceptions.RequestException:
                continue

        # Si aucun endpoint ne fonctionne
        logger.warning("⚠️  Aucune API Ollama détectée, utilisation du mode fallback")
        self.api_type = OllamaAPIType.UNKNOWN

    def generate_text(self,
                      model: str,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.1,
                      max_tokens: int = 1024,
                      timeout: int = 60,
                      ) -> str:
        """Génère du texte avec l'API détectée"""

        if self.api_type == OllamaAPIType.OPENAI_COMPATIBLE:
            return self._generate_openai_compatible(
                model, prompt, system_prompt, temperature, max_tokens, timeout
            )
        elif self.api_type == OllamaAPIType.LEGACY_GENERATE:
            return self._generate_legacy(
                model, prompt, system_prompt, temperature, max_tokens, timeout
            )
        elif self.api_type == OllamaAPIType.LEGACY_CHAT:
            return self._chat_legacy(
                model, prompt, system_prompt, temperature, max_tokens, timeout
            )
        else:
            return self._generate_fallback(
                model, prompt, system_prompt, temperature, max_tokens, timeout
            )

    def _generate_openai_compatible(self, model: str, prompt: str,
                                    system_prompt: Optional[str],
                                    temperature: float, max_tokens: int, timeout: int) -> str:
        """Utilise l'API OpenAI-compatible"""
        url = f"{self.base_url}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            logger.debug(url)
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Format réponse inattendu: {data}")
                return ""

        except Exception as e:
            logger.error(f"Erreur API OpenAI-compatible: {e}")
            return ""

    def _generate_legacy(self, model: str, prompt: str,
                         system_prompt: Optional[str],
                         temperature: float, max_tokens: int, timeout: int) -> str:
        """Utilise l'ancien endpoint /api/generate"""
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

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '')

        except Exception as e:
            logger.error(f"Erreur API legacy generate: {e}")
            return ""

    def _chat_legacy(self, model: str, prompt: str,
                     system_prompt: Optional[str],
                     temperature: float, max_tokens: int, timeout: int) -> str:
        """Utilise l'ancien endpoint /api/chat"""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('message', {}).get('content', '')

        except Exception as e:
            logger.error(f"Erreur API legacy chat: {e}\n{payload}")
            return ""

    def _generate_fallback(self, model: str, prompt: str,
                           system_prompt: Optional[str],
                           temperature: float, max_tokens: int, timeout: int) -> str:
        """Fallback avec package ollama Python"""
        try:
            import ollama

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )

            return response['message']['content']

        except ImportError:
            logger.error("Package 'ollama' non installé")
            return "Échec: API Ollama non détectée et package non installé"
        except Exception as e:
            logger.error(f"Erreur package ollama: {e}")
            return f"Échec: {str(e)}"

    def get_models(self) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles disponibles"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('models', [])

        except Exception as e:
            logger.error(f"Erreur récupération modèles: {e}")
            return []

    def pull_model(self, model: str):
        payload = {
            model: model
        }
        url = f"{self.base_url}/api/pull"

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Erreur API pull model: {e}\n{payload}")
            return ""
