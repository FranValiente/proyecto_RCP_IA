from ultralytics import YOLO
import csv
import torch

weights = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\default_params_tensorboard\all_together.pt"
model = YOLO(weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

video_path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\ya_entrenado\3.mp4'
csv_path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos'

cls_names = {0: 'compresor1', 1: 'compresor2', 2: 'enfermeroT', 3: 'líder',
             4: 'ventilador', 5: 'paciente', 6: 'UCI', 7: 'desconocido'}

print("Iniciando predict")

with open(csv_path + "/predictions_video_3.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "clase", "confianza", "x1", "y1", "x2", "y2"])

    # stream=True: convierte results en generador, libera cada frame tras procesarlo
    for frame_idx, result in enumerate(model.predict(
        source=video_path,
        save=True,
        batch=8,
        classes=[0, 1, 2, 3, 4, 5, 6, 7],
        max_det=8,
        stream=True,
        verbose=True
    )):
        for i, clase in enumerate(result.boxes.cls):
            cls_id = int(clase)
            conf = float(result.boxes.conf[i])
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
            writer.writerow([frame_idx, cls_names[cls_id], round(conf, 4), x1, y1, x2, y2])