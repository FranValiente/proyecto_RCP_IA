import csv
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

# Global variables
FPS = 30

def _get_all_lines(csv_path):
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


def compresor1_info(csv_path):

    frames = _get_all_lines(csv_path)
    best = {}  # {frame_id: (conf, height)}
    for frame in frames[1:]:
        if frame[1] == "compresor1":
            num_frame = int(frame[0])
            conf = float(frame[2])
            y1 = float(frame[4])
            y2 = float(frame[6])
            height = y2 - y1
            if num_frame not in best or conf > best[num_frame][0]:
                best[num_frame] = (conf, height) # Get the highest confidence compresor1 for each frame (if there´s more than one detected)

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
    pct_t_compresion = (total_frames - contadorNone)/total_frames * 100


    ventanas = []
    ventana_actual = []
    inicio_ventana_actual = None
    inicio_primera_ventana = None
    # Partir heights en ventanas separadas por los valores vacíos
    for i, h in enumerate(heights):
        if h is not None:
            if not ventana_actual:  # primer frame de una nueva ventana
                inicio_ventana_actual = i
            ventana_actual.append(h)
        else:
            if ventana_actual and len(ventana_actual) >= 120:
                if inicio_primera_ventana is None:  # solo guardar la primera
                    inicio_primera_ventana = inicio_ventana_actual
                ventanas.append(ventana_actual)
                ventana_actual = []
                inicio_ventana_actual = None

    if ventana_actual and len(ventana_actual) >= 120:
        if inicio_primera_ventana is None:
            inicio_primera_ventana = inicio_ventana_actual
        ventanas.append(ventana_actual)

    return {"alturas": heights, "pct_t_compresion": round(pct_t_compresion,2), "ventanas": ventanas, 't2inicio_comp': round(inicio_primera_ventana/FPS, 2)}


csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_12.csv"

import matplotlib.pyplot as plt
compresiones = compresor1_info(csv_path)
print(f"%t de tiempo con compresiones: {compresiones["pct_t_compresion"]}%")
print(f"Tiempo hasta el inicio de las compresiones: {compresiones["t2inicio_comp"]} s")

heights = compresiones["alturas"]

plt.plot(heights)
plt.ylabel("Altura compresor1 (cm)")
plt.xlabel("Frames")
plt.show()

for ventana in compresiones["ventanas"]:
    ventana = np.array(ventana)
    ventana -= np.min(ventana)

    umbral = 1.5
    media = np.mean(ventana)
    std = np.std(ventana)
    ventana_filtered = np.array([i for i in ventana if abs(i - media) < umbral * std])
    # ventana_filtered = np.where(np.abs(ventana - media) < umbral * std, ventana, media)

    def calcular_metricas(señal, fps):
        picos, _ = find_peaks(señal, prominence=2.5)
        valles, _ = find_peaks([-x for x in señal], prominence=2.5)
        
        duracion = len(señal) / fps
        cpm = (len(picos) / duracion) * 60 if duracion > 0 else 0
        
        diferencias = []
        for p in picos:
            if len(valles) > 0:
                v_cercano = valles[np.argmin(np.abs(valles - p))]
                diff = señal[p] - señal[v_cercano]
                if diff <= 15: diferencias.append(diff) # Control extra para no tener en cuenta en el cálculo de la profundidad puntos con una diferencia excesiva
                
        profundidad = np.mean(diferencias) if diferencias else 0
        return picos, valles, cpm, profundidad

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    plt.subplots_adjust(hspace=0.4)

    # Señal Original
    p1, v1, cpm1, depth1 = calcular_metricas(ventana, FPS)
    ax1.plot(ventana, label=f'Original: {cpm1:.1f} cpm | {depth1:.1f} cm', color='blue', alpha=0.6)
    ax1.plot(p1, ventana[p1], "x", color='red', label="picos")
    ax1.plot(v1, ventana[v1], "x", color='green', label="valles")
    ax1.set_title("Señal Original (Con ruido/outliers)")
    ax1.legend(loc='upper right', fontsize='small')

    # Señal Filtrada
    p2, v2, cpm2, depth2 = calcular_metricas(ventana_filtered, FPS)
    ax2.plot(ventana_filtered, label=f'Procesada: {cpm2:.1f}±10.0 cpm | {depth2:.1f}±1.0 cm', color='darkgreen')
    ax2.plot(p2, ventana_filtered[p2], "x", color='red')
    ax2.plot(v2, ventana_filtered[v2], "x", color='green')
    ax2.set_title(f"Señal Filtrada (Umbral: {umbral}σ)")
    ax2.legend(loc='upper right', fontsize='small')

    plt.show()
#scipy.signal savgol_filter