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