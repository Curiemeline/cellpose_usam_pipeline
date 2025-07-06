import napari
from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton
import os
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

    print("dim: ", mask_data.ndim)
    
    if mask_data.ndim > 2:
        print(f"Le masque a {mask_data.shape[0]} couches.")
        for i in range(mask_data.shape[0]):
            mask = mask_data[i, ...] if mask_data.ndim > 1 else mask_data    # *Expression ternaire* qui prend la i-ème couche du masque si le masque a plus d'une dimension, sinon le garde tel quel
                                                                                    # pour un array (10, 512, 512), ça revient à mask_data[i, :, :]. C'est une ellipse qui prends l’indice i sur le premier axe, et garde les autres dimensions telles quelles
            
            imwrite(save_path.replace(".tif", f"_{i}.tif"), mask)
            print(f"Masque {i} sauvegardé à {save_path}")

    
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

import sys
from qtpy.QtWidgets import QApplication, QFileDialog
from micro_sam.sam_annotator import annotator_2d, _state, annotator_3d, _widgets
import imageio.v3 as imageio
import os
import torch

def launch_2dannotation_viewer(args):
    # Obligatoire avant tout QWidget comme QFileDialog
    app = QApplication(sys.argv)

    # Crée le state
    state = _state.AnnotatorState()

    model_type = "vit_b_lm"  # Type de modèle à utiliser pour les embeddings

    # Étape 1. Demander l'image d'origine
    print("Sélectionnez l'image originale")
    image_path, _ = QFileDialog.getOpenFileName(caption="Sélectionner l'image originale")
    image = imageio.imread(image_path)



    # Étape 2. Calculer les embeddings

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
    add_finetune_button(viewer, args)

    viewer = annotator_2d(
        viewer=viewer,
        image=image,
        segmentation_result=mask,               #  An initial segmentation to load.
                                                # This can be used to correct segmentations with Segment Anything or to save and load progress.
                                                # The segmentation will be loaded as the 'committed_objects' layer.
        embedding_path=embedding_save_path,
        return_viewer=True
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

    viewer = annotator_3d(
        viewer=viewer,
        image=image,
        segmentation_result=mask,               #  An initial segmentation to load.
                                                # This can be used to correct segmentations with Segment Anything or to save and load progress.
                                                # The segmentation will be loaded as the 'committed_objects' layer.
        embedding_path=embedding_save_path,
        return_viewer=True
    )

    napari.run()


if __name__ == "__main__":
    #launch_2dannotation_viewer()
    launch_3dannotation_viewer()
