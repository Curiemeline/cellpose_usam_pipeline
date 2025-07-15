import napari
from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton
from napari.utils.notifications import show_info
import os
from tifffile import imread
import numpy as np
from pathlib import Path
import micro_sam
from micro_sam.sam_annotator.annotator_2d import annotator_2d
print(dir(micro_sam))
from tifffile import imwrite
from Source.finetune import finetune_cellpose, split_dataset

def show_info_message(viewer, message):
    msg = QMessageBox(viewer.window._qt_window)
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle("Information")
    msg.exec_()

# def select_file(title="Select a file"):
#     """Open a QFileDialog and return the selected file path."""
#     filename, _ = QFileDialog.getOpenFileName(caption=title)
#     return filename if filename else None

def save_mask_as_tif(viewer, output_dir):
    if "committed_objects" not in viewer.layers:
        show_info_message(viewer, "Aucune couche 'committed_objects' trouvée.")
        return

    image = viewer.layers["image"].data
    mask_data = viewer.layers["committed_objects"].data

    if mask_data.shape[0] != image.shape[0]:
        show_info_message(viewer, "Le nombre de frames ne correspond pas entre les images et les masques.")
        return

    # # Demande un dossier de sauvegarde
    # output_dir = QFileDialog.getExistingDirectory(caption="Choisir un dossier pour enregistrer les images et masques")
    # if not output_dir:
    #     return
    
    # Force les données en liste de frames pour unifier les cas
    if mask_data.ndim == 2:
        image_frames = [image]
        mask_frames = [mask_data]
    elif mask_data.ndim == 3:
        if mask_data.shape[0] != image.shape[0]:
            show_info_message(viewer, "Le nombre de frames ne correspond pas entre les images et les masques.")
            return
        image_frames = image
        mask_frames = mask_data
    else:
        show_info_message(viewer, f"Format non supporté : ndim = {mask_data.ndim}")
        return

    for i, (img, msk) in enumerate(zip(image_frames, mask_frames)):
        image_filename = os.path.join(output_dir, f"image_{i}.tif")
        mask_filename = os.path.join(output_dir, f"image_{i}_masks.tif")

        imwrite(image_filename, img.astype('uint16'))
        imwrite(mask_filename, msk.astype('uint16'))

        print(f"Frame {i} sauvegardée :")
        print(f"  -> Image : {image_filename}")
        print(f"  -> Masque : {mask_filename}")

    show_info_message(viewer, f"{len(image_frames)} image(s) et masque(s) sauvegardés dans :\n{output_dir}")