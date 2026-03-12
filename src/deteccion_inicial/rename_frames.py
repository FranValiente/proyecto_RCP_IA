import os
import re

def renombrar_frames(directorio):
    # Listar y ordenar archivos para procesar del menor al mayor
    archivos = sorted([f for f in os.listdir(directorio) if f.endswith('.png')])
    
    for nombre_archivo in archivos:
        # Buscar el número en el nombre (ej: 'frame_000001.png')
        match = re.search(r'(\d+)', nombre_archivo)
        if match:
            numero_actual_str = match.group(1)
            numero_nuevo = int(numero_actual_str) - 1
            
            # Mantener el mismo número de ceros a la izquierda (zfill)
            nuevo_numero_str = str(numero_nuevo).zfill(len(numero_actual_str))
            nuevo_nombre = nombre_archivo.replace(numero_actual_str, nuevo_numero_str)
            
            # Rutas completas
            ruta_antigua = os.path.join(directorio, nombre_archivo)
            ruta_nueva = os.path.join(directorio, nuevo_nombre)
            
            # Renombrar usando os.rename
            os.rename(ruta_antigua, ruta_nueva)
            print(f"Renombrado: {nombre_archivo} -> {nuevo_nombre}")

# Uso del script
ruta_carpeta = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\frames_7"
renombrar_frames(ruta_carpeta)