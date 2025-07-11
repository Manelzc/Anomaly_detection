import os
import cv2

input_base = "MyVideos"
output_list = "my_test.txt"
output_base = "X3D_Features"
min_frames = 16

valid_paths = []
invalid_videos = []

print("🔍 Verificando videos...\n")

for label in ['Normal', 'Anomaly']:
    folder = os.path.join(input_base, label)
    if not os.path.exists(folder):
        continue

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mp4"):
            continue

        path = os.path.join(folder, fname)
        cap = cv2.VideoCapture(os.path.abspath(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"📹 {label}/{fname} → {total} frames")

        if total >= min_frames:
            # nombre del archivo de features esperado
            out_name = f"{label}_{os.path.splitext(fname)[0]}.npy"
            feature_path = os.path.join(output_base, out_name).replace("\\", "/")
            valid_paths.append(feature_path)
        else:
            invalid_videos.append((label, fname, total))

# Guardar archivo de test
with open(output_list, "w") as f:
    for path in valid_paths:
        f.write(path + "\n")

print("\n✅ Archivo generado:", output_list)
print(f"👍 Videos válidos: {len(valid_paths)}")
print(f"❌ Videos inválidos (<{min_frames} frames): {len(invalid_videos)}")

if invalid_videos:
    print("\n⚠️ Problemas detectados:")
    for label, fname, n in invalid_videos:
        print(f"   ⛔ {label}/{fname} → {n} frames")