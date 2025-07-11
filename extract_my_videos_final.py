
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda, Normalize, CenterCrop
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import DataLoader
import torch.nn as nn

# ===============================
# Configuración
# ===============================
input_base = "MyVideos"
output_base = "X3D_Features"
os.makedirs(output_base, exist_ok=True)
list_file = "my_test.txt"
model_name = "x3d_l"
frames_per_second = 30

transform_params = {
    "side_size": 320,
    "crop_size": 320,
    "num_frames": 16,
    "sampling_rate": 5,
}

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# ===============================
# Modelo
# ===============================
print("🔄 Cargando modelo X3D-L desde PyTorchVideo...")
model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
model = nn.Sequential(*list(model.blocks[:-1]))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.eval().to(device)

# ===============================
# Transformaciones
# ===============================
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(transform_params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),
        Normalize(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
        Permute((1, 0, 2, 3))
    ])
)

clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

# ===============================
# Preparar videos y paths
# ===============================
all_videos = []
for label in ["Normal", "Anomaly"]:
    folder = os.path.join(input_base, label)
    label_int = 0 if label == "Normal" else 1
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            full_path = os.path.join(folder, file)
            save_path = os.path.join(output_base, label, file.replace(".mp4", ".npy"))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_videos.append((full_path, {
                "label": label_int,
                "video_label": save_path
            }))

dataset = LabeledVideoDataset(
    labeled_video_paths=all_videos,
    clip_sampler=UniformClipSampler(clip_duration),
    transform=transform,
    decode_audio=False
)
loader = DataLoader(dataset, batch_size=1)

label = None
current = None

for inputs in tqdm(loader, desc="Extrayendo features"):
    preds = model(inputs["video"].to(device)).detach().cpu().numpy()
    for i, pred in enumerate(preds):
        if inputs['video_label'][i][:-4] != label:
            if label is not None:
                np.save(label + ".npy", current.squeeze())
            label = inputs['video_label'][i][:-4]
            current = pred[None, ...]
        else:
            current = np.max(np.concatenate((current, pred[None, ...]), axis=0), axis=0)[None, ...]

if label is not None:
    np.save(label + ".npy", current.squeeze())

# ===============================
# Guardar lista my_test.txt
# ===============================
with open(list_file, "w") as f:
    for _, metadata in all_videos:
        f.write(metadata["video_label"].replace("\\", "/") + "\n")

print("✅ Extracción finalizada. Features en", output_base)
