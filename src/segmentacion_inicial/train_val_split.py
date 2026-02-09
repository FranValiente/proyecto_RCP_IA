import random
import shutil
from pathlib import Path

# --- Configuración ---
parent_dir = Path(r"C:\Users\SimIA\Documents\proyecto_RCP_IA\data\finetuning_datasets\segmentaciones_videos_1_5_11")  # <-- carpeta padre
images_dir = parent_dir / "images"
labels_dir = parent_dir / "labels"

train_ratio = 0.80
seed = 230
move_files = True  # True = mover, False = copiar

# extensiones de imagen aceptadas
img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# --- Salidas ---
images_train = images_dir / "train"
images_val   = images_dir / "val"
labels_train = labels_dir / "train"
labels_val   = labels_dir / "val"

for p in [images_train, images_val, labels_train, labels_val]:
    p.mkdir(parents=True, exist_ok=True)

# --- Indexa labels por "stem" (frame_000123) ---
label_files = sorted(labels_dir.glob("*.txt"))
label_by_stem = {p.stem: p for p in label_files}

# --- Encuentra imágenes en images/ (solo nivel raíz) ---
image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
image_by_stem = {p.stem: p for p in image_files}

# --- Empareja por stem ---
common_stems = sorted(set(label_by_stem.keys()) & set(image_by_stem.keys()))
missing_img = sorted(set(label_by_stem.keys()) - set(image_by_stem.keys()))
missing_lbl = sorted(set(image_by_stem.keys()) - set(label_by_stem.keys()))

if missing_img:
    print(f"[WARN] {len(missing_img)} labels sin imagen (ej: {missing_img[:5]})")
if missing_lbl:
    print(f"[WARN] {len(missing_lbl)} imágenes sin label (ej: {missing_lbl[:5]})")

# --- Split reproducible ---
random.seed(seed)  # reproducible [web:18]
random.shuffle(common_stems)  # aleatorio [web:18]

n_total = len(common_stems)
n_train = int(train_ratio * n_total)
train_stems = common_stems[:n_train]
val_stems = common_stems[n_train:]

def transfer(src: Path, dst: Path):
    if move_files:
        shutil.move(str(src), str(dst))  # mover [web:22]
    else:
        shutil.copy2(str(src), str(dst))  # copiar preservando metadatos cuando se puede [web:15]

# --- Ejecuta movimientos/copias ---
for stems, img_out, lbl_out, split_name in [
    (train_stems, images_train, labels_train, "train"),
    (val_stems,   images_val,   labels_val,   "val"),
]:
    for stem in stems:
        img_src = image_by_stem[stem]
        lbl_src = label_by_stem[stem]

        transfer(img_src, img_out / img_src.name)
        transfer(lbl_src, lbl_out / lbl_src.name)

    print(f"{split_name}: {len(stems)} pares")

print("Hecho.")
