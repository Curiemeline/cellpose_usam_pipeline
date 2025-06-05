import subprocess
from tifffile import imread
import numpy as np
from cellpose import models, io


# # def finetune_cellpose(train_dir, pretrained_model, model_name_out,  n_epochs, learning_rate, weight_decay, mask_filter="_seg.npy"):
# #     """Lance le finetuning de Cellpose via la ligne de commande."""
# #     print("Démarrage du finetuning de Cellpose...")
# #     # Construire la commande pour finetuner le modèle
# #     command = [
# #         "python", "-m", "cellpose", "--train",
# #         "--verbose", 
# #         "--dir", train_dir,
# #         #"--test_dir", test_dir,
# #         "--z_axis", "0",
# #         "--min_train_masks", "1",
# #         "--pretrained_model", pretrained_model,
# #         "--n_epochs", str(n_epochs),
# #         "--learning_rate", str(learning_rate),
# #         "--weight_decay", str(weight_decay),
# #         "--chan", "1",                   # Channel for grayscale
# #         "--chan2", "0",                  # Secondary channel
# #         "--model_name_out", model_name_out,
# #         #"--channel_axis", "-1",            # Channel axis
# #         "--mask_filter", mask_filter,
# #         "--use_gpu"  # Ajouter cette option si tu veux utiliser un GPU
# #     ]

# #     # Exécuter la commande
# #     subprocess.run(command)
# #     print("running the cmd")


# # def add_channel_to_data(data):
# #     """
# #     Adapte les données 2D (telles que (nframes, nY, nX)) en ajoutant une dimension pour les channels,
# #     ce qui donne la forme (nframes, nY, nX, 1).

# #     Args:
# #         data (np.array): Image de forme (nframes, nY, nX) représentant un film 2D.

# #     Returns:
# #         np.array: Image modifiée avec une dimension supplémentaire pour les channels, forme (nframes, nY, nX, 1).
# #     """
# #     # Vérifier que la donnée d'entrée a bien la forme attendue (nframes, nY, nX)
# #     if len(data.shape) != 3:
# #         raise ValueError("Les données doivent avoir la forme (nframes, nY, nX).")
    
# #     # Ajouter une dimension pour les channels, ici le canal unique
# #     data_with_channel = np.expand_dims(data, axis=-1)
    
# #     return data_with_channel



# # # Exemple d'utilisation de la fonction
# # if __name__ == "__main__":
# #     train_dir = "/Users/emeline.fratacci/Unit/micro_sam/cellpose_usam_pipeline/Data_output"
# #     #test_dir = "/chemin/vers/ton/dossier/test"
# #     pretrained_model = "cyto3"  # Ou ton propre modèle pré-entraîné
# #     model_name_out = "finetune_model"  # Nom du modèle de sortie

# #     print("Lancement du finetuning de Cellpose...")
# #     img = imread("../Data_output/D2_w1TIRF 491 rEstus_s10_stack.tif")
# #     npy = np.load("../Data_output/D2_w1TIRF 491 rEstus_s10_stack_seg.npy", allow_pickle=True).item()

# #     img_chan = add_channel_to_data(img)
# #     print(img.shape)
# #     print(npy['masks'].shape)
# #     print(img_chan.shape)
# #     #finetune_cellpose(train_dir, pretrained_model, model_name_out, n_epochs=10, learning_rate=0.1, weight_decay=0.0001)


from cellpose import models, io, train
import matplotlib
matplotlib.use('Agg')  # Utiliser 'Agg' pour éviter les conflits avec napari. Sinon quand on fermait la fenêtre napari, le terminal continuait de tourner. 
                        # Pourquoi ? Parce que matplotlib utilise par défaut un backend interactif (souvent Qt5Agg) qui prépare une fenêtre en ARRIERE PLAN quand on crée une figure, ce qu'on fait ici même si on ne fait pas de plt.show() pour la visualiser.
                        # En utilisant le mode non intéractif 'Agg', matplotlib ne tente pas de créer une fenêtre graphique, ce qui évite les conflits avec napari et permet de sauvegarder des figures sans afficher de fenêtre.
import matplotlib.pyplot as plt
import os
import torch

def finetune_cellpose(output_path):
    # train_dir = "/Users/emeline.fratacci/Unit/micro_sam/cellpose_usam_pipeline/Data_output"
    # test_dir = "/Users/emeline.fratacci/Unit/micro_sam/cellpose_usam_pipeline/Data_test"

    output_path = Path(output_path)
    train_dir = output_path.parent / "Train"
    test_dir = output_path.parent / "Test"
    save_dir = output_path.parent.parent / "Models"

    save_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Save directory: {save_dir}")

    output = io.load_train_test_data(str(train_dir), str(test_dir),
                                    mask_filter="_seg.npy", look_one_level_down=False)  # !!! Need to add str to train_dir and test_dir because they are Path objects defined above, and io.load_train_test_data expects strings
    images, labels, image_names, test_images, test_labels, image_names_test = output
    print(images[1].ndim)
    model = models.CellposeModel(gpu=True, model_type='cyto3', pretrained_model='cyto3', diam_mean=80)

    # Avant d'appeler train.train_seg
    X, Y = [], []
    for x, y in zip(images, labels):  # x et y ont chacun shape (83, 512, 512, 2)
        print("x",x.shape)
        print("y",y.shape)
        X.append(x) # oblgiée de faire ça quand j'ai juste un 2D images comme par exemple (400, 400) et pas (83, 512, 512, 2)
        Y.append(y)
        # for t in range(x.shape[0]):
        #     print("xt",x[t].shape)
        #     print("yt",y[t].shape)
        #     X.append(x[t])  # x[t] a shape (512, 512, 2)
        #     Y.append(y[t])  # pareil pour le label
        #     break
        # break




    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=X, train_labels=Y,
        weight_decay=1e-4, SGD=True, learning_rate=0.1,
        n_epochs=60, model_name="my_new_model_wo_cp",
        channel_axis=-1
    )


    # Plot et sauvegarde
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')

    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Chemin de sauvegarde
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"Courbe de loss sauvegardée dans : {save_path}")

    # plt.show()    # potentiel conflit avec napari et qcoreapplication
    plt.close('all')  # Ferme la figure pour libérer la mémoire

import os
import shutil
import random
from pathlib import Path

def split_dataset(finetune_dir, train_ratio=0.8, seed=42):
    random.seed(seed)

    finetune_dir = Path(finetune_dir)
    output_dir = finetune_dir.parent 
    train_dir = output_dir / "Train"    # When working with Path object, we can use / instead of os.path.join
    test_dir = output_dir / "Test"
    
    # Création des dossiers
    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Lister tous les fichiers *_seg.npy (indique une image complète annotée)
    seg_files = list(finetune_dir.glob("*_seg.npy"))
    basenames = [f.stem.replace("_seg", "") for f in seg_files]

    random.shuffle(basenames)
    split_idx = int(len(basenames) * train_ratio)
    train_basenames = basenames[:split_idx]
    test_basenames = basenames[split_idx:]

    def move_files(basenames, dest_dir):
        for base in basenames:
            for ext in [".tif", "_cp_masks.tif", "_seg.npy"]:
                if ext.startswith("_"):
                    filename = f"{base}{ext}"
                else:
                    filename = f"{base}{ext}"
                src = finetune_dir / filename
                dst = dest_dir / filename
                if src.exists():
                    shutil.copy(src, dst)
                else:
                    print(f"Missing File : {src}")

    move_files(train_basenames, train_dir)
    move_files(test_basenames, test_dir)

    print(f"Dataset splitted : {len(train_basenames)} in train, {len(test_basenames)} in test")
    return train_dir, test_dir

