from ultralytics import YOLO
import torch

model = YOLO("yolo26l-pose.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\20260119-102413-v13222-5.mp4'

results = model.predict(source=path,
                        save=True,
                        batch=12,
                        classes=[0], # person
                        max_det=7) # pacient, 5 roles, UCI