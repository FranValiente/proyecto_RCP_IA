from ultralytics import YOLO
import torch

SEED = 27

if __name__ == "__main__":

    model = YOLO(r"C:\Users\SimIA\Documents\proyecto_RCP_IA\yolo26x.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)


    #-------------------------------------------- DEFAULT --------------------------------------------
    # Finetuning
    results = model.train(
        data=r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\all_videos\data.yaml",
        epochs=300,
        project=r"C:\Users\SimIA\Documents\proyecto_RCP_IA\runs\detect\new_run",
        name="train_all_together",
        seed=SEED,
        device = 0,
        patience = 30,
        cache=True,
        imgsz=640,
        batch=8,
        save=True, # Guarda pesos (por defecto)
        plots=True # Gráficas de métricas
    )