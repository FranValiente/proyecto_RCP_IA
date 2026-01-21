from ultralytics import YOLO
import torch

model = YOLO("yolo26n-pose.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\20260119-102413-v13222-5.mp4'

results = model(path)
print(type(results))
print(results)