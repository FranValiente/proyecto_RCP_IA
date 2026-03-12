import os
import shutil
from pathlib import Path


def get_max_frame_number(directory: Path) -> int:
    """Devuelve el número de frame más alto encontrado en un directorio."""
    max_num = -1
    for f in directory.iterdir():
        if f.stem.startswith("frame_"):
            try:
                num = int(f.stem.split("_")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                continue
    return max_num


def merge_yolo_datasets(dataset_paths: list, output_path: str):
    """
    Fusiona N datasets YOLO en uno solo renombrando los frames para evitar
    colisiones de nombres.

    Args:
        dataset_paths : Lista de rutas a los datasets de entrada (orden importa).
        output_path   : Ruta del dataset fusionado de salida.
    """
    output = Path(output_path)
    splits = ["train", "val"]

    # Crear estructura de carpetas de salida
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Procesando split: {split}")
        print(f"{'='*50}")

        offset = 0  # Offset acumulado para este split

        for ds_idx, ds_path in enumerate(dataset_paths):
            ds_path = Path(ds_path)
            images_dir = ds_path / "images" / split
            labels_dir = ds_path / "labels" / split

            if not images_dir.exists():
                print(f"  [WARN] No existe: {images_dir}. Saltando.")
                continue

            # Ordenar frames por número para consistencia
            image_files = sorted(
                [f for f in images_dir.iterdir() if f.stem.startswith("frame_")],
                key=lambda f: int(f.stem.split("_")[1])
            )

            if not image_files:
                print(f"  [WARN] Sin frames en {images_dir}. Saltando.")
                continue

            count_ok = 0
            count_missing_label = 0

            for img_file in image_files:
                original_num = int(img_file.stem.split("_")[1])
                new_num = original_num + offset
                new_stem = f"frame_{new_num:06d}"
                ext = img_file.suffix

                # Copiar imagen con nuevo nombre
                dst_img = output / "images" / split / f"{new_stem}{ext}"
                shutil.copy2(img_file, dst_img)

                # Copiar label con nuevo nombre (si existe)
                src_label = labels_dir / f"{img_file.stem}.txt"
                if src_label.exists():
                    dst_label = output / "labels" / split / f"{new_stem}.txt"
                    shutil.copy2(src_label, dst_label)
                    count_ok += 1
                else:
                    count_missing_label += 1
                    print(f"  [WARN] Label no encontrado: {src_label.name}")

            # El offset del siguiente dataset = frame más alto de este dataset + 1
            max_frame = int(image_files[-1].stem.split("_")[1])
            print(f"  Dataset {ds_idx + 1} ({ds_path.name}): "
                  f"{len(image_files)} imágenes copiadas "
                  f"(offset aplicado: {offset}, rango: "
                  f"frame_{offset:06d} → frame_{offset + max_frame:06d})")
            if count_missing_label > 0:
                print(f"    ⚠ {count_missing_label} labels no encontrados.")

            offset += max_frame + 1  # Actualizar offset para el siguiente dataset

    print(f"\n✅ Merge completado. Dataset fusionado en: {output.resolve()}")


if __name__ == "__main__":
    
    DATASET_PATHS = [
        r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_1_5_11",
        r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_3_6_7",
        r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_4_8_13",
    ]
    OUTPUT_PATH = r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets"

    merge_yolo_datasets(DATASET_PATHS, OUTPUT_PATH)
