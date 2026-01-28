from ultralytics import YOLO
import torch
import os

model = YOLO(r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\YOLO26_finetuned_4_8.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\6.mp4'

results = model.predict(source=path,
                        save=True,
                        batch=12,
                        classes=[0,1,2,3,4,5,6,7], # roles
                        max_det=8) # pacient, 5 roles, UCI, desconocido