from ultralytics import YOLO
import torch
import os

weights = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\default_params_tensorboard\all_together.pt"
model = YOLO(weights)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\2.mp4' # 2 12, testear 9 10 gineco

print("Iniciando predict")
results = model.predict(source=path,
                        save=True,
                        batch=8,
                        classes=[0,1,2,3,4,5,6,7], # roles
                        max_det=8, # pacient, 5 roles, UCI, desconocido
                        verbose=True)