import os
import shutil

# ffmpeg -i input.mp4 nombre_carpeta/frame_%06d.png

# --- Configuración ---
labels_dir = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_1_5_11\labels\train"
output_images = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_1_5_11\images\train"

frames_video1 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_11"
frames_video2 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_1"
frames_video3 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_5"  # <-- ajusta

# Rangos globales (incluyentes) + carpeta de frames correspondiente
VIDEOS = [
    {"name": "v1", "global_start": 0,    "global_end": 1669, "frames_dir": frames_video1},
    {"name": "v2", "global_start": 1670, "global_end": 6983, "frames_dir": frames_video2},
    {"name": "v3", "global_start": 6984, "global_end": 10713, "frames_dir": frames_video3},
]

IMG_EXT = ".png"

os.makedirs(output_images, exist_ok=True)  # crea carpeta destino si no existe [web:7]

txt_files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))

def find_video_for_frame(frame_num: int):
    for v in VIDEOS:
        if v["global_start"] <= frame_num <= v["global_end"]:
            return v
    return None

for txt_file in txt_files:
    frame_num = int(txt_file.split("_")[-1].replace(".txt", ""))

    v = find_video_for_frame(frame_num)
    if v is None:
        print(f"¡ERROR! Frame global {frame_num:06d} fuera de rangos (archivo {txt_file})")
        continue

    # Mapea global -> local (0-based dentro del vídeo)
    local_frame_num = frame_num - v["global_start"]
    local_frame_name = f"frame_{local_frame_num:06d}{IMG_EXT}"
    source_path = os.path.join(v["frames_dir"], local_frame_name)

    # El nombre destino mantiene el índice global para que cuadre con labels
    target_name = txt_file.replace(".txt", IMG_EXT)
    target_path = os.path.join(output_images, target_name)

    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)  # conserva metadatos cuando es posible [web:15]
        print(f"[{v['name']}] Copiado {source_path} → {target_path}")
    else:
        print(f"¡ERROR! No existe {source_path} (para {txt_file}, global {frame_num:06d}, local {local_frame_num:06d})")

print("¡Listo! Dataset preparado.")
