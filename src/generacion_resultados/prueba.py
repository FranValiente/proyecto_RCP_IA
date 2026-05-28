import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from metricas.metricas import compresor1_info, t_asign_roles, pct_t_roles
from generacion_resultados.generate_document import generar_reporte_pdf

csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_5.csv"
compresiones = compresor1_info(csv_path)
heights = compresiones["alturas"]
t2rol = t_asign_roles(csv_path)
pct_roles = pct_t_roles(csv_path)

FPS = 29

logo_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\logosimia.jpg"
output_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\informe.pdf"

generar_reporte_pdf(compresiones, t2rol, 
                    pct_roles, heights, 
                    FPS, output_path, logo_path)