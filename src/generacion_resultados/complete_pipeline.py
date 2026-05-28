import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from metricas.generate_csv import generate_csv
from generacion_resultados.generate_document import generar_reporte_pdf
from metricas.metricas import compresor1_info, t_asign_roles, pct_t_roles

FPS = 29

LOGO_SIMIA = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\logosimia.jpg"

video = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\videos\2.mp4"
results_name = "prueba"
csv_save_dir = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\generacion_resultados"
predict = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\generacion_resultados"
csv = generate_csv(video_path=video, csv_name=results_name, csv_path=csv_save_dir, video_save_dir=predict)

compresiones = compresor1_info(csv)
heights = compresiones["alturas"]
t2rol = t_asign_roles(csv)
pct_roles = pct_t_roles(csv)

nombre_pdf = "reporte.pdf"
generar_reporte_pdf(compresiones, t2rol, pct_roles, heights, FPS, csv_save_dir+nombre_pdf, LOGO_SIMIA)