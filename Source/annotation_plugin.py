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

def save_mask_as_tif(viewer):
    if "committed_objects" not in viewer.layers:
        show_info_message(viewer, "Aucune couche 'committed_objects' trouvée.")
        return

    mask_data = viewer.layers["committed_objects"].data

    save_path, _ = QFileDialog.getSaveFileName(caption="Enregistrer le masque", filter="tif files (*.tif)")
    if not save_path:
        return

    if not save_path.endswith(".tif"):
        save_path += ".tif"

    imwrite(save_path, mask_data)
    show_info_message(viewer, f"Masque sauvegardé au format Cellpose :\n{save_path}")

def add_save_button(viewer):
    btn = QPushButton("Sauver masque pour Cellpose")
    btn.clicked.connect(lambda: save_mask_as_tif(viewer))
    viewer.window.add_dock_widget(btn, area='right')


def on_finetune_button_clicked(viewer, args):
    split_dataset(finetune_dir=args.output)
    finetune_cellpose(output_path=args.output)
    show_info_message(viewer, "Finetuning terminé. Modèle sauvegardé dans le dossier 'Models'.")


def add_finetune_button(viewer, args):
    btn_finetune = QPushButton("Lancer le finetuning")
    btn_finetune.clicked.connect(lambda: on_finetune_button_clicked(viewer, args))
    viewer.window.add_dock_widget(btn_finetune, area='right')


# def launch_annotation_viewer():
#     viewer = napari.Viewer()

#     # Ajouter manuellement le widget micro-sam s'il est bien installé
#     try:
#         viewer.window.add_plugin_dock_widget(
#             plugin_name="micro-sam",
#             widget_name="Annotator 2d"
#         )
#         show_info_message(viewer, "Widget Micro-SAM ajouté avec succès.")
#     except Exception as e:
#         show_info_message(viewer,f"Erreur lors de l'ajout du widget Micro-SAM : {e}")

#     # Supprimer la couche 'committed_objects' si elle existe
#     if "committed_objects" in viewer.layers:
#         viewer.layers.remove("committed_objects")
#         show_info_message(viewer,"Layer 'committed_objects' supprimée.")

#     # Sélection de l’image brute
#     raw_path = select_file("Sélectionne l'image RAW")
#     if raw_path:
#         raw = imread(raw_path)
#         viewer.add_image(raw, name=Path(raw_path).stem)

#     ################ WIP
#     annotator_2d(
#         image=raw, # <- image brute
#         embedding_path="Embeddings/test_vit_l_lm.zarr",  # <- embeddings seront automatiquement calculés
#         model_type="vit_l_lm",
#         tile_shape=(256, 256),
#         halo=(32, 32),
#         return_viewer=False,
#     )
#     ################ WIP


    
#     # Sélection du masque à renommer
#     mask_path = select_file("Sélectionne le MASK à annoter")
#     if mask_path:
#         mask = imread(mask_path)

#         # Convertit en labels (si pas déjà entier)
#         if mask.dtype != np.int32 and mask.dtype != np.uint16:
#             mask = mask.astype(np.uint16)

#         # Ajoute le masque comme 'committed_objects' avec type Labels
#         viewer.add_labels(mask, name="committed_objects")

#     # Ajoute le bouton de sauvegarde
#     add_save_button(viewer)

#     napari.run()

import napari
import sys
from qtpy.QtWidgets import QApplication, QFileDialog
from micro_sam.sam_annotator import annotator_2d, _state, annotator_3d
import imageio.v3 as imageio
import os
import torch

def launch_2dannotation_viewer():
    # Obligatoire avant tout QWidget comme QFileDialog
    app = QApplication(sys.argv)

    # Crée le state
    state = _state.AnnotatorState()

    model_type = "vit_b_lm"  # Type de modèle à utiliser pour les embeddings

    print("hoy")
    # Étape 1. Demander l'image d'origine
    print("Sélectionnez l'image originale")
    image_path, _ = QFileDialog.getOpenFileName(caption="Sélectionner l'image originale")
    image = imageio.imread(image_path)



    # Étape 2. Calculer les embeddings
    #embedding_save_path = os.path.splitext(image_path)[0] + "_embedding.zarr"

    # Récupérer le chemin vers le dossier parent de l'image
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)

    # Créer le dossier "Embeddings" au même niveau
    embedding_dir = os.path.join(parent_dir, "Embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    # Nom de fichier d'embedding basé sur le nom de l’image
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    embedding_save_path = os.path.join(embedding_dir, f"{image_basename}_embedding_{model_type}.zarr")





    state.initialize_predictor(
        image,
        model_type=model_type,
        save_path=embedding_save_path,
        ndim=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Étape 3. Demander le masque existant
    print("Sélectionnez le masque associé")
    mask_path, _ = QFileDialog.getOpenFileName(caption="Sélectionner le masque (optionnel)")
    mask = imageio.imread(mask_path) if mask_path else None

    # Étape 4. Lancer le viewer avec tout préchargé
    viewer = napari.Viewer()
    add_save_button(viewer)

    annotator_2d(
        viewer=viewer,
        image=image,
        segmentation_result=mask,               #  An initial segmentation to load.
                                                # This can be used to correct segmentations with Segment Anything or to save and load progress.
                                                # The segmentation will be loaded as the 'committed_objects' layer.
        embedding_path=embedding_save_path,
    )


    napari.run()
    

def launch_3dannotation_viewer(args):
    # Obligatoire avant tout QWidget comme QFileDialog
    # app = QApplication.instance()
    # if app is None:
    app = QApplication(sys.argv)

    # Crée le state
    state = _state.AnnotatorState()

    model_type = "vit_b_lm"  # Type de modèle à utiliser pour les embeddings


    # Étape 1. Demander l'image d'origine
    print("Sélectionnez l'image originale")
    image_path, _ = QFileDialog.getOpenFileName(caption="Sélectionner l'image originale")
    image = imageio.imread(image_path)



    # Étape 2. Calculer les embeddings
    #embedding_save_path = os.path.splitext(image_path)[0] + "_embedding.zarr"

    # Récupérer le chemin vers le dossier parent de l'image
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)

    # Créer le dossier "Embeddings" au même niveau
    embedding_dir = os.path.join(parent_dir, "Embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    # Nom de fichier d'embedding basé sur le nom de l’image
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    embedding_save_path = os.path.join(embedding_dir, f"{image_basename}_embedding_{model_type}.zarr")





    state.initialize_predictor(
        image,
        model_type=model_type,
        save_path=embedding_save_path,
        ndim=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Étape 3. Demander le masque existant
    print("Sélectionnez le masque associé")
    mask_path, _ = QFileDialog.getOpenFileName(caption="Sélectionner le masque (optionnel)")
    mask = imageio.imread(mask_path) if mask_path else None

    # Étape 4. Lancer le viewer avec tout préchargé
    viewer = napari.Viewer()
    add_save_button(viewer)
    add_finetune_button(viewer, args)

    annotator_3d(
        viewer=viewer,
        image=image,
        segmentation_result=mask,               #  An initial segmentation to load.
                                                # This can be used to correct segmentations with Segment Anything or to save and load progress.
                                                # The segmentation will be loaded as the 'committed_objects' layer.
        embedding_path=embedding_save_path,
    )


    napari.run()



if __name__ == "__main__":
    #launch_2dannotation_viewer()
    launch_3dannotation_viewer()
