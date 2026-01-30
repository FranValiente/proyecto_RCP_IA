from ultralytics import YOLO
import torch

if __name__ == "__main__":

    model = YOLO("yolo26x.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Finetuning
    results = model.train(
        data=r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\segmentaciones_videos_4_8_13\data.yaml",
        epochs=100,
        device = 0,
        imgsz=640,
        batch=16,
        save=True, # Guarda pesos (por defecto)
        plots=True # Gráficas de métricas
    )