#!/bin/bash
set -e

# Mettre à jour les paquets
sudo apt-get update

# Installer les dépendances
sudo apt-get install -y curl

# Ajouter le dépôt NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Installer le toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Redémarrer Docker
sudo systemctl restart docker



# Démarre Ollama
ollama serve &

# Attendre le démarrage du service
until curl -s http://localhost:11434/api/tags > /dev/null; do
  echo "⏳ Attente du démarrage d’Ollama..."
  sleep 2
done

# Téléchargement des modèles
ollama pull llama3:8b:q4_K_M
ollama pull mxbai-embed-large
killall ollama || true

# Garder le conteneur actif
wait
