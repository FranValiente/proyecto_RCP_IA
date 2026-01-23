from ultralytics import YOLO
import torch
import os

model = YOLO("yolo26l-pose.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

path = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\8.mp4'

# results = model.predict(source=path,
#                             save=True,
#                             batch=12,
#                             classes=[0], # person
#                             max_det=7) # pacient, 5 roles, UCI
trackerpath = r'C:\Users\SimIA\Documents\proyecto_RCP_IA\src\segmentacion_inicial\botsort_adapted.yaml'
results = model.track(source=path,tracker=trackerpath,save=True,classes=[0],batch=12,max_det=7)