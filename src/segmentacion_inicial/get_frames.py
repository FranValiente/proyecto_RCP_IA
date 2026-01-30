import os
import shutil

# --- Configuración ---
labels_dir = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\segmentaciones_videos_4_8_13\labels\train"
output_images = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\dataset\images"

frames_video1 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_8"
frames_video2 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_4"
frames_video3 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_13"  # <-- ajusta

# Rangos globales (incluyentes) + carpeta de frames correspondiente
# Vídeo 1: frame_000000 a frame_002514
# Vídeo 2: frame_002524 a frame_005249
# Vídeo 3: frame_005281 a frame_008447
VIDEOS = [
    {"name": "v1", "global_start": 0,    "global_end": 2514, "frames_dir": frames_video1},
    {"name": "v2", "global_start": 2524, "global_end": 5249, "frames_dir": frames_video2},
    {"name": "v3", "global_start": 5281, "global_end": 8447, "frames_dir": frames_video3},
]

IMG_EXT = ".png"  # cambia a ".jpg" si aplica

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
