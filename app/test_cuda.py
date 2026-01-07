import os
import subprocess

import torch

print("=== Diagnostic CUDA/PyTorch ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")

# Vérifier les variables d'environnement
print(f"\n=== Variables d'environnement ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Non défini')}")
print(f"PATH contenant CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")

# Vérifier nvidia-smi
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(f"\n=== nvidia-smi output ===")
    if result.returncode == 0:
        print("✅ nvidia-smi fonctionne")
        print(result.stdout[:500])  # Premières 500 caractères
    else:
        print(f"❌ nvidia-smi échoue: {result.stderr}")
except FileNotFoundError:
    print("❌ nvidia-smi non trouvé (CUDA Toolkit non installé ou pas dans PATH)")
