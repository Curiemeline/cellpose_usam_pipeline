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
    print(output_path)
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
                                    mask_filter="_masks",look_one_level_down=False)  # !!! Need to add str to train_dir and test_dir because they are Path objects defined above, and io.load_train_test_data expects strings
    images, labels, image_names, test_images, test_labels, image_names_test = output

    print(len(test_labels))
    print([np.unique(lbl) for lbl in test_labels])
    print([lbl.shape for lbl in test_labels])

    # print(images[1].ndim)
    model = models.CellposeModel(gpu=True, pretrained_model='cpsam')

    # # Avant d'appeler train.train_seg
    # X, Y = [], []
    # for x, y in zip(images, labels):  # x et y ont chacun shape (83, 512, 512, 2)
    #     # print("x",x.shape)
    #     # print("y",y.shape)
    #     X.append(x) # oblgiée de faire ça quand j'ai juste un 2D images comme par exemple (400, 400) et pas (83, 512, 512, 2)
    #     Y.append(y)
    #     # for t in range(x.shape[0]):
    #     #     print("xt",x[t].shape)
    #     #     print("yt",y[t].shape)
    #     #     X.append(x[t])  # x[t] a shape (512, 512, 2)
    #     #     Y.append(y[t])  # pareil pour le label
    #     #     break
    #     # break


    model_name = "cpsam_100ep_lrdef_30im"

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images, train_labels=labels,
        test_data=test_images, test_labels=test_labels, 
        min_train_masks=1,
        #weight_decay=1e-4,
        #learning_rate=0.1,
        n_epochs=100, model_name=model_name,
        channel_axis=None # None pour 2D, -1 pour 3D (mais pas de 3D dans ce cas)
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
    save_path = os.path.join(save_dir, f"loss_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Courbe de loss sauvegardée dans : {save_path}")

    # plt.show()    # potentiel conflit avec napari et qcoreapplication
    plt.close('all')  # Ferme la figure pour libérer la mémoire

import os
import shutil
import random
from pathlib import Path

import shutil
import random
from pathlib import Path

def split_dataset(finetune_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    finetune_dir = Path(finetune_dir)
    output_dir = finetune_dir.parent
    train_dir = output_dir / "Train"
    test_dir = output_dir / "Test"

    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Trouver les paires image/masque
    image_files = [f for f in finetune_dir.glob("*.tif") if "_masks" not in f.name]
    mask_files = list(finetune_dir.glob("*_masks.tif"))

    pairs = []
    for img in image_files: 
        name = img.stem  # ex: image_1
        expected_mask = finetune_dir / f"{name}_masks.tif"  # We rebuild the path to the expected mask file to see if for example image_1_masks.tif exists
        if expected_mask.exists():                          # This is more robust than using ```matching_masks = [m for m in mask_files if name in m.name]```, because image_1 is also IN image_10, so it would match image_10_masks.tif as well and image_1 won't appear in the list of files, hence finetuning will crash bc of missing mask.
            pairs.append((img, expected_mask))
        else:
            print(f"Aucun masque trouvé pour {img.name}")

    # Shuffle + split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)   
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    def copy_pairs(pairs, dest_dir):
        for img_path, mask_path in pairs:
            print(f"Processing pair: {img_path.name} and {mask_path.name}")
            for f in [img_path, mask_path]:
                dst = dest_dir / f.name
                print(f"Copying {f} → {dst}")
                shutil.copy(f, dst)

    copy_pairs(train_pairs, train_dir)
    copy_pairs(test_pairs, test_dir)

    print(f"Dataset split: {len(train_pairs)} train, {len(test_pairs)} test")
    return train_dir, test_dir



# # def split_dataset(finetune_dir, train_ratio=0.8, seed=42):
# #     random.seed(seed)
# #     print("FINETUNE")
# #     finetune_dir = Path(finetune_dir)   # TODO Supposed to be output directory. Calling the variable finetune_dir might be misleading. 
# #     output_dir = finetune_dir.parent    
# #     train_dir = output_dir / "Train"    # When working with Path object, we can use / instead of os.path.join
# #     test_dir = output_dir / "Test"

# #     print(f"Finetune directory: {finetune_dir}")
# #     print(f"Train directory: {train_dir}")
# #     print(f"Test directory: {test_dir}")
    
    
# #     # Création des dossiers
# #     for d in [train_dir, test_dir]:
# #         d.mkdir(parents=True, exist_ok=True)

# #     # Lister tous les fichiers *_seg.npy 
# #     seg_files = list(finetune_dir.glob("*_seg.npy"))
# #     basenames = [f.stem.replace("_seg", "") for f in seg_files]

# #     random.shuffle(basenames)
# #     split_idx = int(len(basenames) * train_ratio)
# #     train_basenames = basenames[:split_idx]
# #     test_basenames = basenames[split_idx:]

# #     def move_files(basenames, dest_dir):
# #         for base in basenames:
# #             for ext in [".tif", "_cp_masks.tif", "_seg.npy"]:
# #                 if ext.startswith("_"):
# #                     filename = f"{base}{ext}"
# #                 else:
# #                     filename = f"{base}{ext}"
# #                 src = finetune_dir / filename
# #                 dst = dest_dir / filename
# #                 if src.exists():
# #                     shutil.copy(src, dst)
# #                 else:
# #                     print(f"Missing File : {src}")

# #     move_files(train_basenames, train_dir)
# #     move_files(test_basenames, test_dir)

# #     print(f"Dataset splitted : {len(train_basenames)} in train, {len(test_basenames)} in test")
# #     return train_dir, test_dir

if __name__ == "__main__":
    finetune_cellpose("D:\micro_sam\Datasets\Output")