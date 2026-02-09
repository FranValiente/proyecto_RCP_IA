from ultralytics import YOLO
import torch

if __name__ == "__main__":

    model = YOLO(r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\yolo_finetuning_4_8_13.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Finetuning
    results = model.train(
        data=r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_1_5_11\data.yaml",
        epochs=100,
        device = 0,
        patience = 7,
        workers = 0,
        imgsz=640,
        batch=16,
        save=True, # Guarda pesos (por defecto)
        plots=True # Gráficas de métricas
    )