#!/bin/bash

echo "🚀 Démarrage d'Ollama..."

# Démarrer le serveur en arrière-plan
/bin/ollama serve &
SERVER_PID=$!
echo "✅ Serveur Ollama started with PID ${SERVER_PID}"

# Fonction pour vérifier si le serveur est prêt
wait_for_server() {
    echo "⏳ Attente du démarrage du serveur..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q "200"; then
            echo "✅ Serveur Ollama prêt après ${attempt}s"
            return 0
        fi
        echo "  Tentative $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "❌ Le serveur Ollama n'a pas démarré à temps"
    return 1
}

# Attendre le serveur
if wait_for_server; then
    # Télécharger les modèles depuis la variable d'environnement
    if [ -n "$MY_OLLAMA_MODELS" ]; then
        echo "📥 Téléchargement des modèles spécifiés..."
        echo "    $MY_OLLAMA_MODELS"
        IFS=',' read -ra MODELS <<< "$MY_OLLAMA_MODELS"
        for model in "${MODELS[@]}"; do
            echo "  → Téléchargement de: $model"
            if /bin/ollama pull "$model"; then
                echo "    ✅ $model téléchargé avec succès"
            else
                echo "    ⚠️  Échec du téléchargement de $model"
            fi
        done
    fi

    # Télécharger les modèles par défaut
    DEFAULT_MODELS=("llama3" "codellama")
    for model in "${DEFAULT_MODELS[@]}"; do
        if ! /bin/ollama list | grep -q "$model"; then
            echo "📥 Téléchargement du modèle par défaut: $model"
            /bin/ollama pull "$model" || echo "⚠️  Échec pour $model"
        fi
    done
    kill $SERVER_PID

    echo "🎉 Initialisation terminée!"
#    /bin/ollama serve
#    echo "📡 API disponible sur: http://localhost:11434"

    # Attendre que le serveur continue de tourner
#    wait $SERVER_PID
else
    echo "❌ Échec de l'initialisation"
    exit 1
fi
