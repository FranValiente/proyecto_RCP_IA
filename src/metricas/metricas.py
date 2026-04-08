import csv
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

# Global variables
FPS = 29 # YOLO predict gives a .avi video of 29 fps

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

    for i, h in enumerate(heights):
        if h is not None:
            if not ventana_actual:
                inicio_ventana_actual = i
            ventana_actual.append(h)
        else:
            if ventana_actual:
                ventanas.append({
                    "alturas_raw": ventana_actual,
                    "frame_inicio": inicio_ventana_actual,
                    "frame_fin": i - 1  # i es el primer None, el frame anterior es el fin
                })
                ventana_actual = []
                inicio_ventana_actual = None
    if ventana_actual:
        ventanas.append({
            "alturas_raw": ventana_actual,
            "frame_inicio": inicio_ventana_actual,
            "frame_fin": len(heights) - 1
        })

    GAP_MAXIMO = 15

    ventanas_fusionadas = [ventanas[0]] if ventanas else []
    for ventana in ventanas[1:]:
        anterior = ventanas_fusionadas[-1]
        gap = ventana["frame_inicio"] - anterior["frame_fin"]
        if gap <= GAP_MAXIMO:
            # Rellenar el gap con None y fusionar
            gap_nones = [None] * gap
            anterior["alturas_raw"] = anterior["alturas_raw"] + gap_nones + ventana["alturas_raw"]
            anterior["frame_fin"] = ventana["frame_fin"]
        else:
            ventanas_fusionadas.append(ventana)

    ventanas = ventanas_fusionadas

    # Filtrar ventanas demasiado cortas (menos de N frames)
    MIN_FRAMES = 5
    ventanas = [v for v in ventanas if len(v["alturas_raw"]) >= MIN_FRAMES]

    inicio_primera_ventana = ventanas[0]['frame_inicio']/FPS

    UMBRAL = 1.5
    for v in ventanas:
        signal = pd.Series(v["alturas_raw"]).interpolate(method='linear').to_numpy()
        signal = np.array(signal, dtype=float)
        signal -= np.min(signal)
        media = np.mean(signal)
        std = np.std(signal)
        signal_filtrada = np.where(np.abs(signal - media) < UMBRAL * std, signal, media)
        v["alturas_filtradas"] = signal_filtrada  # nueva key agregada al diccionario de las ventanas: alturas normalizadas + outliers sustituidos por la media

    return {
        "alturas": heights, 
        "pct_t_compresion": round(pct_t_compresion,2), 
        "ventanas": ventanas, 
        't2inicio_comp': round(inicio_primera_ventana, 2)
    }


def calcular_metricas_compresiones(signal, fps):
    picos, _ = find_peaks(signal, prominence=2.5)
    valles, _ = find_peaks([-x for x in signal], prominence=2.5)
    
    duracion = len(signal) / fps
    cpm = (len(picos) / duracion) * 60 if duracion > 0 else 0
    
    diferencias = []
    for p in picos:
        if len(valles) > 0:
            v_cercano = valles[np.argmin(np.abs(valles - p))]
            diff = signal[p] - signal[v_cercano]
            if diff <= 15: diferencias.append(diff) # Control extra para no tener en cuenta en el cálculo de la profundidad puntos con una diferencia excesiva
            
    profundidad = np.mean(diferencias)-1 if diferencias else 0
    return picos, valles, cpm, profundidad


def t_asign_roles(csv_path):
    """
    Calcula el tiempo hasta la primera detección estable de cada rol,
    definida como la primera vez que el rol aparece durante al menos
    MIN_SEGUNDOS de forma continua (tolerando gaps de hasta GAP_MAXIMO frames).

    Returns:
        dict {rol: tiempo_en_segundos} — None si el rol no alcanza el mínimo.
    """
    ROLES = ["ventilador", "líder", "compresor2", "enfermeroT"]
    MIN_SEGUNDOS = 3
    MIN_FRAMES = MIN_SEGUNDOS * FPS   # 87 frames a 29fps
    GAP_MAXIMO = 15

    frames = _get_all_lines(csv_path)

    # Recoger todos los frames en que aparece cada rol
    apariciones = {rol: set() for rol in ROLES}
    for frame in frames[1:]:
        clase = frame[1]
        if clase in apariciones:
            apariciones[clase].add(int(frame[0]))

    t2rol = {}

    for rol in ROLES:
        frames_rol = sorted(apariciones[rol])

        if not frames_rol:
            t2rol[rol] = None
            continue

        # Agrupar en segmentos continuos tolerando gaps <= GAP_MAXIMO
        segmentos = []
        seg_inicio = frames_rol[0]
        seg_fin    = frames_rol[0]

        for f in frames_rol[1:]:
            if f - seg_fin <= GAP_MAXIMO:
                seg_fin = f          # el gap se absorbe, el segmento continúa
            else:
                segmentos.append((seg_inicio, seg_fin))
                seg_inicio = f
                seg_fin    = f
        segmentos.append((seg_inicio, seg_fin))  # último segmento

        # Primer segmento cuya duración en frames supera el mínimo
        primera_asignacion = None
        for seg_inicio, seg_fin in segmentos:
            if (seg_fin - seg_inicio + 1) >= MIN_FRAMES:
                primera_asignacion = seg_inicio
                break

        t2rol[rol] = round(primera_asignacion / FPS, 2) if primera_asignacion is not None else None

    return t2rol

csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_7.csv"
t2rol = t_asign_roles(csv_path)

print(t2rol)

# import matplotlib.pyplot as plt
# compresiones = compresor1_info(csv_path)

# heights = compresiones["alturas"]

# plt.plot(heights)
# plt.ylabel("Altura compresor1 (cm)")
# plt.xlabel("Frames")
# plt.show()
# print('='*60)
# print(f"Información temporal de la calidad de las compresiones:")
# print(f"El paciente ha estado un {compresiones["pct_t_compresion"]}% del vídeo recibiendo compresiones")
# print(f"Tiempo hasta el inicio de las compresiones: {compresiones["t2inicio_comp"]} s")
# print(f'Información sobre cada intervalo de compresiones continuadas:')

# for window in compresiones["ventanas"]:
#     ventana_filtered = window["alturas_filtradas"]
#     # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
#     # plt.subplots_adjust(hspace=0.4)

#     # # signal Original
#     # p1, v1, cpm1, depth1 = calcular_metricas(ventana, FPS) # picos, valles, compresiones por minuto, profundidad compresiones
#     # ax1.plot(ventana, label=f'Original: {cpm1:.1f} cpm | {depth1:.1f} cm', color='blue', alpha=0.6)
#     # ax1.plot(p1, ventana[p1], "x", color='red', label="picos")
#     # ax1.plot(v1, ventana[v1], "x", color='green', label="valles")
#     # ax1.set_title("signal Original (Con ruido/outliers)")
#     # ax1.legend(loc='upper right', fontsize='small')

#     # # signal Filtrada
#     p2, v2, cpm2, depth2 = calcular_metricas_compresiones(ventana_filtered, FPS)

#     t_inicio = window['frame_inicio']/FPS # En segundos
#     t_fin = window['frame_fin']/FPS
#     minutos_inicio, segundos_inicio = divmod(t_inicio, 60)
#     minutos_fin, segundos_fin = divmod(t_fin, 60)

#     print(f"    Intervalo entre {int(minutos_inicio)}min:{segundos_inicio:.1f}s - {int(minutos_fin)}min:{segundos_fin:.1f}s: {cpm2:.1f}±10.0 cpm, {depth2:.1f}±1.0 cm")
#     # ax2.plot(ventana_filtered, label=f'Procesada: {cpm2:.1f}±10.0 cpm | {depth2:.1f}±1.0 cm', color='darkgreen')
#     # ax2.plot(p2, ventana_filtered[p2], "x", color='red')
#     # ax2.plot(v2, ventana_filtered[v2], "x", color='green')
#     # ax2.set_title(f"signal Filtrada (Umbral: {umbral}σ)")
#     # ax2.legend(loc='upper right', fontsize='small')

#     # plt.show()
# #scipy.signal savgol_filter