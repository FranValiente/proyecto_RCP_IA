from ultralytics import YOLO
import csv
import torch
import os

weights = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\default_params_tensorboard\all_together.pt"
model = YOLO(weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

video_path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\ya_entrenado\7.mp4' # 2 12, testear 9 10 gineco
csv_path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos'

print("Iniciando predict")
results = model.predict(source=video_path,
                        save=True,
                        batch=8,
                        classes=[0,1,2,3,4,5,6,7], # roles
                        max_det=8, # pacient, 5 roles, UCI, desconocido
                        verbose=True)

# print(result.boxes.cls)
# print(result.boxes.conf)
# result.names es un dict con la etiqueta que se le corresponde a cada clase {0: 'compresor1', 1: 'compresor2', 2: 'enfermeroT', 3: 'líder', 4: 'ventilador', 5: 'paciente', 6: 'UCI', 7: 'desconocido'}
dict = {0: 'compresor1', 1: 'compresor2', 2: 'enfermeroT', 3: 'líder', 4: 'ventilador', 5: 'paciente', 6: 'UCI', 7: 'desconocido'}
with open(csv_path+"/predictions_video_7.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "clase", "confianza", "x1", "y1", "x2", "y2"])

    for frame_idx, result in enumerate(results):
        for i, clase in enumerate(result.boxes.cls):
            cls_id = int(clase)
            conf = float(result.boxes.conf[i])
            x1, y1, x2, y2 = result.boxes.xyxy[i].tolist() # tolist() because it´s a Tensor
            writer.writerow([frame_idx, dict[cls_id], round(conf, 4), x1, y1, x2, y2])