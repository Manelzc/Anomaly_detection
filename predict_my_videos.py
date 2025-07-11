import argparse
import torch
import numpy as np
import os
from model import Model

# Argumentos por línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument("--test_rgb_list", type=str, required=True, help="Archivo con lista de paths de test")
parser.add_argument("--model_arch", type=str, default="fast", help="Arquitectura del modelo (fast, base, tiny)")
parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de batch")
args = parser.parse_args()

# Configuración
MODEL_PATH = {
    "fast": "saved_models/913base.pkl",
    "tiny": "saved_models/888tiny.pkl"
}

assert args.model_arch in MODEL_PATH, f"Modelo no soportado: {args.model_arch}"

# Cargar modelo
model = Model()
model.load_state_dict(torch.load(MODEL_PATH[args.model_arch], map_location="cpu"))
model.eval()

# Leer archivos
with open(args.test_rgb_list, "r") as f:
    test_files = [line.strip() for line in f.readlines()]

print("\n📊 Resultados de predicción:\n")

for path in test_files:
    if not os.path.isfile(path):
        print(f"❌ No se encontró el archivo: {path}")
        continue

    feat = np.load(path)

    if feat.ndim == 1:
        feat = feat.reshape(1, -1)

    x = torch.from_numpy(feat).float()  # [N, D]
    with torch.no_grad():
        score_tensor, _ = model(x.unsqueeze(0))  # [1, N, 1]
        score = torch.sigmoid(score_tensor).mean().item()

    label = "Anomalía" if score > 0.5 else "Normal"
    print(f"{os.path.basename(path)} → Predicción: {label} ({score:.2f})")