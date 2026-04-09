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


def _segmentar_rol(frames_rol, gap_maximo=15):
    """
    Agrupa una lista ordenada de frames en segmentos continuos,
    tolerando gaps de hasta gap_maximo frames entre detecciones.

    Returns:
        Lista de tuplas (seg_inicio, seg_fin).
    """
    if not frames_rol:
        return []

    segmentos = []
    seg_inicio = frames_rol[0]
    seg_fin    = frames_rol[0]

    for f in frames_rol[1:]:
        if f - seg_fin <= gap_maximo:
            seg_fin = f
        else:
            segmentos.append((seg_inicio, seg_fin))
            seg_inicio = f
            seg_fin    = f
    segmentos.append((seg_inicio, seg_fin))

    return segmentos


def _primer_segmento_estable(segmentos, min_frames):
    """
    Devuelve el frame de inicio del primer segmento que cumple
    la duración mínima, o None si ninguno la cumple.
    """
    for seg_inicio, seg_fin in segmentos:
        if (seg_fin - seg_inicio + 1) >= min_frames:
            return seg_inicio
    return None


def t_asign_roles(csv_path, min_segundos=3):
    ROLES = ["ventilador", "líder", "compresor2", "enfermeroT"]
    MIN_FRAMES = min_segundos * FPS
    GAP_MAXIMO = 15

    frames = _get_all_lines(csv_path)
    apariciones = {rol: set() for rol in ROLES}
    for frame in frames[1:]:
        if frame[1] in apariciones:
            apariciones[frame[1]].add(int(frame[0]))

    t2rol = {}
    for rol in ROLES:
        frames_rol = sorted(apariciones[rol])
        segmentos  = _segmentar_rol(frames_rol, GAP_MAXIMO)
        frame_asig = _primer_segmento_estable(segmentos, MIN_FRAMES)
        t2rol[rol] = round(frame_asig / FPS, 2) if frame_asig is not None else None

    return t2rol


def pct_t_roles(csv_path, min_segundos=3): # Tras asignación
    ROLES = ["compresor1", "ventilador", "líder", "compresor2", "enfermeroT"]
    MIN_FRAMES = min_segundos * FPS
    GAP_MAXIMO = 15

    frames = _get_all_lines(csv_path)
    apariciones = {rol: set() for rol in ROLES}
    max_frame_video = 0

    for frame in frames[1:]:
        num_frame = int(frame[0])
        max_frame_video = max(max_frame_video, num_frame)
        if frame[1] in apariciones:
            apariciones[frame[1]].add(num_frame)

    pct_roles = {}
    for rol in ROLES:
        frames_rol = sorted(apariciones[rol])
        segmentos  = _segmentar_rol(frames_rol, GAP_MAXIMO)
        frame_asig = _primer_segmento_estable(segmentos, MIN_FRAMES)

        if frame_asig is None:
            pct_roles[rol] = None
            continue

        frames_tras_asig = sum(1 for f in frames_rol if f >= frame_asig)
        total_tras_asig  = max_frame_video - frame_asig + 1
        pct_roles[rol]   = round((frames_tras_asig / total_tras_asig) * 100, 2)

    return pct_roles


csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_6.csv"
compresiones = compresor1_info(csv_path)
heights = compresiones["alturas"]
t2rol = t_asign_roles(csv_path)
pct_roles = pct_t_roles(csv_path)

min_t2ventilador, seg_t2ventilador = divmod(t2rol['ventilador'], 60)
min_t2lider, seg_t2lider = divmod(t2rol['líder'], 60)
min_t2compresor2, seg_t2compresor2 = divmod(t2rol['compresor2'], 60)
min_t2enfermeroT, seg_t2enfermeroT = divmod(t2rol['enfermeroT'], 60)
min_t2compresor1, seg_t2compresor1 = divmod(compresiones["t2inicio_comp"], 60)

import matplotlib.pyplot as plt

plt.plot(heights)
plt.ylabel("Altura compresor1 (cm)")
plt.xlabel("Frames")
plt.show()
print('='*60)
print(f"Tiempos hasta la asignación de los roles:")
print(f"    -Tiempo hasta asignación del compresor1: {int(min_t2compresor1)}min:{seg_t2compresor1:.1f}s")
print(f"    -Tiempo hasta asignación del ventilador: {int(min_t2ventilador)}min:{seg_t2ventilador:.1f}s")
print(f"    -Tiempo hasta asignación del líder: {int(min_t2lider)}min:{seg_t2lider:.1f}s")
print(f"    -Tiempo hasta asignación del compresor2: {int(min_t2compresor2)}min:{seg_t2compresor2:.1f}s")
print(f"    -Tiempo hasta asignación del enfermeroT: {int(min_t2enfermeroT)}min:{seg_t2enfermeroT:.1f}s")
print('='*60)
print(f"Porcentaje de tiempo que ha estado en su rol (una vez se ha establecido):")
print(f"    -Compresor1: {pct_roles["compresor1"]}%")
print(f"    -Ventilador: {pct_roles["ventilador"]}%")
print(f"    -Líder: {pct_roles["líder"]}%")
print(f"    -Compresor2: {pct_roles["compresor2"]}%")
print(f"    -EnfermeroT: {pct_roles["enfermeroT"]}%")
print('='*60)
print(f"Información temporal de la calidad de las compresiones:")
print(f"El paciente ha estado un {compresiones["pct_t_compresion"]}% del vídeo recibiendo compresiones")
print(f"Tiempo hasta el inicio de las compresiones: {int(min_t2compresor1)}min:{seg_t2compresor1:.1f}s")
print(f'Información sobre cada intervalo de compresiones continuadas:')
for window in compresiones["ventanas"]:
    ventana_filtered = window["alturas_filtradas"]
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    # plt.subplots_adjust(hspace=0.4)

    # # signal Original
    # p1, v1, cpm1, depth1 = calcular_metricas(ventana, FPS) # picos, valles, compresiones por minuto, profundidad compresiones
    # ax1.plot(ventana, label=f'Original: {cpm1:.1f} cpm | {depth1:.1f} cm', color='blue', alpha=0.6)
    # ax1.plot(p1, ventana[p1], "x", color='red', label="picos")
    # ax1.plot(v1, ventana[v1], "x", color='green', label="valles")
    # ax1.set_title("signal Original (Con ruido/outliers)")
    # ax1.legend(loc='upper right', fontsize='small')

    # # signal Filtrada
    p2, v2, cpm2, depth2 = calcular_metricas_compresiones(ventana_filtered, FPS)

    t_inicio = window['frame_inicio']/FPS # En segundos
    t_fin = window['frame_fin']/FPS
    minutos_inicio, segundos_inicio = divmod(t_inicio, 60)
    minutos_fin, segundos_fin = divmod(t_fin, 60)

    print(f"    -Compresiones realizadas entre {int(minutos_inicio)}min:{segundos_inicio:.1f}s - {int(minutos_fin)}min:{segundos_fin:.1f}s: frecuencia={cpm2:.1f}±10.0 cpm; profundidad={depth2:.1f}±1.0 cm")
    # ax2.plot(ventana_filtered, label=f'Procesada: {cpm2:.1f}±10.0 cpm | {depth2:.1f}±1.0 cm', color='darkgreen')
    # ax2.plot(p2, ventana_filtered[p2], "x", color='red')
    # ax2.plot(v2, ventana_filtered[v2], "x", color='green')
    # ax2.set_title(f"signal Filtrada (Umbral: {umbral}σ)")
    # ax2.legend(loc='upper right', fontsize='small')

    # plt.show()
print('='*60)
# #scipy.signal savgol_filter