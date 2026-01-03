# Dockerfile simple - utilise MY_OLLAMA_MODELS variable d'environnement
FROM ollama/ollama:latest

COPY init-model.sh /
COPY init-base.sh /
RUN chmod +x /init-model.sh
RUN chmod +x /init-base.sh
RUN /init-base.sh
RUN /init-model.sh

# Définir les modèles à télécharger via variable d'environnement

#ENV MY_OLLAMA_MODELS=("llama3","deepseek-coder:33b","deepseek-coder:1.3b","codellama")

EXPOSE 11434

#ENTRYPOINT ["/bin/bash","/init.sh"]

