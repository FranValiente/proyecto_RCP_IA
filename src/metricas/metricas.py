import csv
from scipy.signal import find_peaks
import numpy as np
from scipy.fft import fft

FPS = 30

def get_all_lines(csv_path):
    """
    Lee un archivo CSV y devuelve todas sus filas como una lista de listas.

    Args:
        csv_path (str): La ruta del archivo CSV que se desea leer.

    Returns:
        list: Una lista donde cada elemento es otra lista que representa 
              una fila del archivo CSV.
    """

    with open(csv_path, "r", newline="") as csvfile:
        roles_per_frame = csv.reader(csvfile)

        lines = [line for line in roles_per_frame]

    return lines


def picos_ventana(ventana):
    señal = ventana  # lista de floats

    indices_picos, _  = find_peaks(señal)
    indices_valles, _ = find_peaks([-x for x in señal])  # invertir para encontrar mínimos

    picos  = [señal[i] for i in indices_picos]
    valles = [señal[i] for i in indices_valles]

    return {
        "picos": picos,
        "valles": valles,
        "media_picos": sum(picos) / len(picos) if picos else None,
        "media_valles": sum(valles) / len(valles) if valles else None,
    }


def profundidad_compresiones(csv_path):

    frames = get_all_lines(csv_path)
    best = {}  # {frame_id: (conf, height)}
    for frame in frames[1:]:
        if frame[1] == "compresor1":
            num_frame = int(frame[0])
            conf = float(frame[2])
            y1 = float(frame[4])
            y2 = float(frame[6])
            height = y2 - y1
            if num_frame not in best or conf > best[num_frame][0]:
                best[num_frame] = (conf, height)

    max_frame = max(best.keys()) if best else -1
    heights = []
    contadorNone = 0
    for f in range(max_frame+1):
        if f in best:
            heights.append(0.36*best[f][1]) # 0.36 es el factor que relaciona una altura de compresor1 conocida en px con su altura en cm
        else:
            heights.append(None)
            contadorNone += 1
    
    total_frames = (max_frame + 1)
    pct_compresor1 = (total_frames - contadorNone)/total_frames * 100

    # Partir heights en ventanas separadas por los valores vacíos
    ventanas = []
    ventana_actual = []
    for h in heights:
        if h is not None:
            ventana_actual.append(h)
        else:
            if ventana_actual and len(ventana_actual) >= 120: # evita añadir ventanas vacías si hay Nones consecutivos
                ventanas.append(ventana_actual)
                ventana_actual = []
    if ventana_actual and len(ventana_actual) >= 120: # última ventana si el vídeo termina con señal
        ventanas.append(ventana_actual)

    return {"alturas": heights, "pct_compresor1": round(pct_compresor1,2), "ventanas": ventanas}

csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_3.csv"

import matplotlib.pyplot as plt
compresiones = profundidad_compresiones(csv_path)
print(f"%t del compresor 1: {compresiones["pct_compresor1"]}%")

heights = compresiones["alturas"]

plt.plot(heights)
plt.ylabel("Altura compresor1 (cm)")
plt.xlabel("Frames")
plt.show()

for ventana in compresiones["ventanas"]:
    ventana = np.array(ventana)
    ventana -= np.min(ventana)

    indices_picos, _  = find_peaks(ventana, prominence=1.5)
    indices_valles, _ = find_peaks([-x for x in ventana], prominence=1.5)  # invertir para encontrar mínimos

    duracion_segundos = len(ventana) / FPS
    cpm_temporal = (len(indices_picos) / duracion_segundos) * 60

    diferencias = []
    for idx_pico in indices_picos:
        if len(indices_valles) == 0:
            break
        # Valle más cercano a este pico
        idx_valle_cercano = indices_valles[np.argmin(np.abs(indices_valles - idx_pico))]
        diff = ventana[idx_pico] - ventana[idx_valle_cercano]
        if diff <= 30:  # descartar outliers
            diferencias.append(diff)

    profundidad_media = np.mean(diferencias) if diferencias else 0

    plt.plot(ventana, label=f'Parámetros de las compresiones: f={cpm_temporal:.2f}; depth={profundidad_media:.2f}')
    plt.plot(indices_picos,  ventana[indices_picos],  "x", label="picos")
    plt.plot(indices_valles, ventana[indices_valles], "x", label="valles")
    plt.legend()
    plt.show()