import os
import shutil
from pathlib import Path

# Configuración: ajusta estos valores
labels_dir = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\segmentaciones_videos_4_8\labels\train"  # Tu carpeta con frame_XXXXXX.txt
output_images = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\dataset\images"
frames_video1 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_8"
frames_video2 = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_4"



# Mapeo manual de rangos (Ajusta según tus números reales)
# Vídeo 1: frame_000000 a frame_002512
# Vídeo 2: frame_002524 a frame_005249 (por ejemplo)
video1_end = 2515  # frame_002513 sería el siguiente
video2_start = 2524

os.makedirs(output_images, exist_ok=True)

# Lista todos los .txt
txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
txt_files.sort()

for txt_file in txt_files:
    frame_num = int(txt_file.split('_')[-1].replace('.txt', ''))
    
    if frame_num <= video1_end:
        # Frame del vídeo 1
        source_dir = frames_video1
        # Asume que el índice coincide: frame_002512.txt → frame_002512.png en video1
        source_file = txt_file.replace('.txt', '.png')  # o .png
        source_path = os.path.join(source_dir, source_file)
    else:
        # Frame del vídeo 2
        source_dir = frames_video2
        # Mapea frame global a frame local del vídeo 2
        local_frame_num = frame_num - video2_start
        local_frame_name = f"frame_{local_frame_num:06d}.png"  # Ajusta formato
        source_path = os.path.join(source_dir, local_frame_name)
    
    # Copia imagen al output
    if os.path.exists(source_path):
        target_path = os.path.join(output_images, txt_file.replace('.txt', '.png'))
        shutil.copy2(source_path, target_path)
        print(f"Copiado {source_path} → {target_path}")
    else:
        print(f"¡ERROR! No existe {source_path} para {txt_file}")

print("¡Listo! Dataset preparado.")
