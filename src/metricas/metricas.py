import csv

def get_roles_per_frame(csv_path):
    """_summary_

    Args:
        csv_file (_type_): _description_

    Returns:
        _type_: _description_
    """

    with open(csv_path, "r", newline="") as csvfile:
        roles_per_frame = csv.reader(csvfile)

        for line in roles_per_frame:
            print(line)

    return


def profundidad_compresiones():

    return

csv_path = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\src\metricas\predicciones_videos\predictions_video_2.csv"
get_roles_per_frame(csv_path)