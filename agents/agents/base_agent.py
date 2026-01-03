import logging
from abc import ABC
from typing import Dict, Any, Optional

from agents.ollama_client import AdaptiveOllamaClient
from vector.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def analyze(self, context: Dict[str, Any], vector_store: VectorStore) -> Dict[str, Any]:
        pass

    def __init__(self, config: Dict[str, Any], agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.agent_config = self._get_agent_config()

        try:
            ollama_config = config.get('ollama', {})
            base_url = ollama_config.get('base_url', 'http://localhost:11434')
            self.ollama_client = AdaptiveOllamaClient(base_url)
            agents = config.get('analysis', {}).get('agents', {})
            agent =agents.get(agent_name)
            if agent:
                model_name = agent.get('model', 'llama2')
                models = self.ollama_client.get_models()
                existe = any(m['model'] == model_name or m['name'] == model_name for m in models)
                if not existe:
                    logger.debug(f"✅ Pull model '{model_name}'")
                    pull_model = self.ollama_client.pull_model(model_name)
                    logger.info(f"✅ Load model {model_name} = {pull_model}")
                logger.info(f"✅ Agent '{agent_name}' initialisé avec {model_name}")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Ollama: {e}")
            self.ollama_client = None

    def _get_agent_config(self) -> Dict[str, Any]:
        """Récupère la configuration spécifique de l'agent"""
        agents_config = self.config.get('analysis', {}).get('agents', {})

        # Configuration par défaut
        default_config = {
            'model': 'llama2',
            'temperature': 0.1,
            'max_tokens': 4096,
            'timeout': 60
        }

        # Fusionner avec la config spécifique de l'agent
        agent_specific = agents_config.get(self.agent_name, {})
        return {**default_config, **agent_specific}

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> str:
        if not self.ollama_client:
            return ""

        temp = self.agent_config.get('temperature', temperature)
        tok = self.agent_config.get('max_tokens', 2048)
        tout = self.agent_config.get('timeout', 60)
        llm = self.agent_config.get('model', 'llama2')
        return self.ollama_client.generate_text(
            model=llm,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=tok,
            timeout=tout
        )

    def _get_timestamp(self) -> str:
        """Retourne un timestamp formaté"""
        from datetime import datetime
        return datetime.now().isoformat()
