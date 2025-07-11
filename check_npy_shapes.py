import os
import numpy as np

# Ruta base donde están las features
base_dir = "X3D_Features"
expected_shape = (192,)  # <- Cambia esto si se usa otro modelo

print("🔍 Verificando archivos .npy en:", base_dir)

valid = []
invalid = []

for fname in sorted(os.listdir(base_dir)):
    if not fname.endswith(".npy"):
        continue

    path = os.path.join(base_dir, fname)
    try:
        data = np.load(path)
        if data.shape == expected_shape:
            valid.append(fname)
        else:
            invalid.append((fname, data.shape))
    except Exception as e:
        print(f"❌ {fname}: ERROR al cargar - {e}")
        invalid.append((fname, "error"))

# Resultados
print(f"\n✅ Features válidas: {len(valid)}")
print(f"❌ Features corruptas o mal formadas: {len(invalid)}")

if invalid:
    print("\n⚠️ Archivos con shape incorrecto:")
    for name, shape in invalid:
        print(f"   ⛔ {name} → shape = {shape}")
