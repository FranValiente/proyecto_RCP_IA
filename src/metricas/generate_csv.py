from ultralytics import YOLO
import csv
import torch

weights = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\weights\default_params_tensorboard\all_together.pt"
cls_names = {0: 'compresor1', 1: 'compresor2', 2: 'enfermeroT', 3: 'líder',
             4: 'ventilador', 5: 'paciente', 6: 'UCI', 7: 'desconocido'}

def generate_csv(video_path, csv_name, csv_path, video_save_dir):
    """Procesa un video frame a frame utilizando un modelo YOLO para detectar roles de RCP (IDs del 0 al 7). 
    Guarda el video resultante con las detecciones visuales y exporta los datos detallados de cada detección 
    (número de frame, nombre de la clase, confianza y coordenadas de la caja delimitadora) en un archivo CSV formateado.

    Args:
        video_path (str | Path): ruta al vídeo a predecir
        csv_name (str): nombre del archivo del .csv generado, sin el formato de archivo
        csv_path (str | Path): ruta del csv generado
        video_save_dir (_tystr | Pathpe_): ruta del vídeo con las predicción de YOLO.
    """

    model = YOLO(weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    relative_csv_name = "/" + csv_name + ".csv"

    print("Iniciando predict")
    with open(csv_path + relative_csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "clase", "confianza", "x1", "y1", "x2", "y2"])

        # stream=True: convierte results en generador, libera cada frame tras procesarlo
        for frame_idx, result in enumerate(model.predict(
            source=video_path,
            save=True,
            project=video_save_dir,
            batch=8,
            classes=[0, 1, 2, 3, 4, 5, 6, 7],
            max_det=8,
            stream=True,
            verbose=True
        )):
            for i, clase in enumerate(result.boxes.cls):
                cls_id = int(clase)
                conf = float(result.boxes.conf[i])
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                writer.writerow([frame_idx, cls_names[cls_id], round(conf, 4), x1, y1, x2, y2])

    return csv_path + relative_csv_name