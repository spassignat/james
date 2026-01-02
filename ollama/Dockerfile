# Dockerfile - basé sur l'image officielle Ollama
FROM ollama/ollama:latest


USER root

# Installer des utilitaires utiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates jq && \
    rm -rf /var/lib/apt/lists/*

# Démarrer temporairement le serveur Ollama pendant le build pour tirer les modèles
RUN bash -c "\
    ollama serve & \
    sleep 5 && \
    ollama pull llama3 \
    ollama pull deepseek-coder:33b \
    killall ollama || true \
"


# && \
#    ollama pull codellama && \
#    ollama pull mxbai-embed-large && \
# Exposer le port API
EXPOSE 11434

# Point d'entrée (l'image officielle gère déjà l'exec)
#ENTRYPOINT ["/usr/bin/ollama"]
#CMD ["serve"]
