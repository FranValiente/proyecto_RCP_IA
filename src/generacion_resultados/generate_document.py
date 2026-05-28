import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib.pyplot as plt
from fpdf import FPDF
from metricas.metricas import calcular_metricas_compresiones

def generar_reporte_pdf(compresiones, t2rol, pct_roles, heights, FPS, output_path="reporte_rcp.pdf", logo_path=None):

    temp_img = "temp_plot.png"
    plt.figure(figsize=(10, 4))
    plt.plot(heights)
    plt.ylabel("Altura compresor1 (cm)")
    plt.xlabel("Frames")
    plt.title("Evolución de las alturas de compresión")
    plt.tight_layout()
    plt.savefig(temp_img) # Save the plot as a temporal image
    plt.close()

    min_t2ventilador, seg_t2ventilador = divmod(t2rol['ventilador'], 60)
    min_t2lider, seg_t2lider = divmod(t2rol['líder'], 60)
    min_t2compresor2, seg_t2compresor2 = divmod(t2rol['compresor2'], 60)
    min_t2enfermeroT, seg_t2enfermeroT = divmod(t2rol['enfermeroT'], 60)
    min_t2compresor1, seg_t2compresor1 = divmod(compresiones["t2inicio_comp"], 60)

    pdf = FPDF()
    pdf.add_page()
    
    if logo_path and os.path.exists(logo_path):
        pdf.image(logo_path, x=160, y=10, w=30)
    
    pdf.set_y(20)

    def write_line(text, font_style='', size=11, height=7):
        pdf.set_x(10)
        pdf.set_font("Helvetica", style=font_style, size=size)
        pdf.multi_cell(0, height, text)

    write_line("Informe del Análisis de Simulación RCP", font_style='B', size=16, height=10)
    pdf.ln(5)

    # Insert the plot
    pdf.image(temp_img, x=10, w=180)
    pdf.ln(5)

    write_line("Tiempos hasta la asignación de los roles:", font_style='B')
    write_line(f"    - Tiempo hasta asignación del compresor1: {int(min_t2compresor1)}min:{seg_t2compresor1:.1f}s")
    write_line(f"    - Tiempo hasta asignación del ventilador: {int(min_t2ventilador)}min:{seg_t2ventilador:.1f}s")
    write_line(f"    - Tiempo hasta asignación del líder: {int(min_t2lider)}min:{seg_t2lider:.1f}s")
    write_line(f"    - Tiempo hasta asignación del compresor2: {int(min_t2compresor2)}min:{seg_t2compresor2:.1f}s")
    write_line(f"    - Tiempo hasta asignación del enfermeroT: {int(min_t2enfermeroT)}min:{seg_t2enfermeroT:.1f}s")
    pdf.ln(5)

    write_line("Porcentaje de tiempo que ha estado en su rol (una vez establecido):", font_style='B')
    write_line(f"    - Compresor1: {pct_roles['compresor1']}%")
    write_line(f"    - Ventilador: {pct_roles['ventilador']}%")
    write_line(f"    - Líder: {pct_roles['líder']}%")
    write_line(f"    - Compresor2: {pct_roles['compresor2']}%")
    write_line(f"    - EnfermeroT: {pct_roles['enfermeroT']}%")
    pdf.ln(5)

    write_line("Información temporal de la calidad de las compresiones:", font_style='B')
    write_line(f"El paciente ha estado un {compresiones['pct_t_compresion']}% del vídeo recibiendo compresiones.")
    write_line(f"Tiempo hasta el inicio de las compresiones: {int(min_t2compresor1)}min:{seg_t2compresor1:.1f}s")
    write_line("Información sobre cada intervalo de compresiones continuadas:", font_style='I')

    for window in compresiones["ventanas"]:
        ventana_filtered = window["alturas_filtradas"]
        
        _, _, cpm2, depth2 = calcular_metricas_compresiones(ventana_filtered, FPS)

        t_inicio = window['frame_inicio'] / FPS
        t_fin = window['frame_fin'] / FPS
        minutos_inicio, segundos_inicio = divmod(t_inicio, 60)
        minutos_fin, segundos_fin = divmod(t_fin, 60)

        interval_text = f"    - Compresiones entre {int(minutos_inicio)}min:{segundos_inicio:.1f}s - {int(minutos_fin)}min:{segundos_fin:.1f}s: frecuencia={cpm2:.1f}±10.0 cpm; profundidad={depth2:.1f}±1.0 cm"
        write_line(interval_text)

    pdf.output(output_path) # Generate the final PDF

    if os.path.exists(temp_img): # Clear the temporal image file
        os.remove(temp_img)
        
    print(f"Informe generado exitosamente en: {output_path}")

