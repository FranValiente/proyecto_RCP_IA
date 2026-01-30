from ultralytics import YOLO
import torch
import os

model = YOLO(r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\yolo_finetuning_4_8_13.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\7.mp4'

results = model.predict(source=path,
                        save=True,
                        batch=12,
                        classes=[0,1,2,3,4,5,6,7], # roles
                        max_det=8) # pacient, 5 roles, UCI, desconocido