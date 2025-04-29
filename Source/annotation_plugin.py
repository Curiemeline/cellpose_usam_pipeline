import napari
from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton
from napari.utils.notifications import show_info
import os
from tifffile import imread
import numpy as np
from pathlib import Path

def show_info_message(viewer, message):
    msg = QMessageBox(viewer.window._qt_window)
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle("Information")
    msg.exec_()

def select_file(title="Select a file"):
    """Open a QFileDialog and return the selected file path."""
    filename, _ = QFileDialog.getOpenFileName(caption=title)
    return filename if filename else None

def save_mask_as_npy(viewer):
    if "committed_objects" not in viewer.layers:
        show_info_message(viewer, "Aucune couche 'committed_objects' trouvée.")
        return

    mask_data = viewer.layers["committed_objects"].data

    save_path, _ = QFileDialog.getSaveFileName(caption="Enregistrer le masque", filter="NumPy files (*.npy)")
    if not save_path:
        return

    if not save_path.endswith(".npy"):
        save_path += ".npy"

    np.save(save_path, {"masks": mask_data})
    show_info_message(viewer, f"Masque sauvegardé au format Cellpose :\n{save_path}")

def add_save_button(viewer):
    btn = QPushButton("💾 Sauver masque pour Cellpose")
    btn.clicked.connect(lambda: save_mask_as_npy(viewer))
    viewer.window.add_dock_widget(btn, area='right')

def launch_annotation_viewer():
    viewer = napari.Viewer()

    # Ajouter manuellement le widget micro-sam s'il est bien installé
    try:
        viewer.window.add_plugin_dock_widget(
            plugin_name="micro-sam",
            widget_name="Annotator 2d"
        )
        show_info_message(viewer, "Widget Micro-SAM ajouté avec succès.")
    except Exception as e:
        show_info_message(viewer,f"Erreur lors de l'ajout du widget Micro-SAM : {e}")

    # Supprimer la couche 'committed_objects' si elle existe
    if "committed_objects" in viewer.layers:
        viewer.layers.remove("committed_objects")
        show_info_message(viewer,"Layer 'committed_objects' supprimée.")

    # Sélection de l’image brute
    raw_path = select_file("Sélectionne l'image RAW")
    if raw_path:
        raw = imread(raw_path)
        viewer.add_image(raw, name=Path(raw_path).stem)

    # Sélection du masque à renommer
    mask_path = select_file("Sélectionne le MASK à annoter")
    if mask_path:
        mask = imread(mask_path)

        # Convertit en labels (si pas déjà entier)
        if mask.dtype != np.int32 and mask.dtype != np.uint16:
            mask = mask.astype(np.uint16)

        # Ajoute le masque comme 'committed_objects' avec type Labels
        viewer.add_labels(mask, name="committed_objects")

    # Ajoute le bouton de sauvegarde
    add_save_button(viewer)

    napari.run()

if __name__ == "__main__":
    launch_annotation_viewer()
