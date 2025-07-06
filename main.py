import argparse
from Source.segmentation import run_cellpose_cli, extract_grads_gray, tracking_centroids
from Source.crops import generate_random_crops, crop_img
from Source.annotation_plugin import launch_2dannotation_viewer, launch_3dannotation_viewer
from Source.finetune import finetune_cellpose, split_dataset
import numpy as np
import os
import shutil
import glob
from tifffile import imwrite, imread

from cellpose import models

def launch_annotator3d(args):
    """
    Function to set the mode to 2D annotator in micro-sam.
    """

    # Créer le chemin vers le dossier Stack (dans le dossier parent)
    parent_dir = os.path.abspath(os.path.join(args.output, os.pardir))
    stack_dir = os.path.join(parent_dir, "Stack")
    os.makedirs(stack_dir, exist_ok=True)

    if args.crop == False:
        print("no cropping, loading images from input directory...")
        image_files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".tif", ".tiff", ".TIF", ".TIFF"))
        ])
        
        img_list = []
        for img_path in image_files:
            print(img_path)
            img = imread(str(img_path))
            img_list.append(img)
        
        stack = np.stack(img_list, axis=0)  # Stack the images along a new dimension
        print("stacked original images")

    if args.crop:
        print("Cropping images before launching 2D annotator...")
        rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
        print(f"Generated {len(rfiles)} random crops from the input directory.")
        cropped_img = crop_img(sorted(rfiles), output_dir=args.output, size=args.crop_size)

        stack = np.stack(cropped_img, axis=0)  # Stack the cropped images along a new dimension
        
        
    if args.segment:
        print("Segmenting...")
        print(f"Input folder for segmentation: {args.output}")

        if args.crop == False:
            print("No cropping, copying original images from input directory to output directory...")
            patterns = args.pattern
            print("patterns:", patterns)
            if patterns is None or len(patterns) == 0:
                patterns = [""]
            print("Copying original images from input directory to output directory...")
            # Copier tous les fichiers depuis args.input vers args.output
            for filename in os.listdir(args.input):
                if all(p in filename for p in patterns):
                    # Vérifier si le fichier correspond au modèle
                    src_path = os.path.join(args.input, filename)
                    dst_path = os.path.join(args.output, filename)
                    if os.path.isfile(src_path):  # Ignore les sous-dossiers
                        shutil.copy(src_path, dst_path)

        print("Segmenting images from output directory...")
        run_cellpose_cli(input_folder=args.output, model_type=args.model, custom_model=args.custom_model, diameter=args.diameter)
        extract_grads_gray(input_folder=args.output)
        

        mask_files = sorted(glob.glob(os.path.join(args.output, "*_cp_masks.tif")))
        print(f"Found {len(mask_files)} mask files for stacking.")
        masks = [imread(m) for m in mask_files]
        print(f"Number of masks loaded: {len(masks)}, with total shape: {masks[0].shape}")

        grads_files = sorted(glob.glob(os.path.join(args.output, "*_gradsXY_gray.tif")))
        print(f"Found {len(grads_files)} gradient files for stacking.")
        grads = [imread(g) for g in grads_files]
        print(f"Number of gradients loaded: {len(grads)}, with total shape: {grads[0].shape}")

        
        # Sauvegarde
        save_path_mask = os.path.join(stack_dir, "masks_stacked.tif")
        imwrite(save_path_mask, np.stack(masks, axis=0), imagej=True)

        save_path_grads = os.path.join(stack_dir, "grads_stacked.tif")
        imwrite(save_path_grads, np.stack(grads, axis=0))

        if args.tracking:
            tracking_centroids(input_folder=stack_dir)
            
            tracked_files = sorted(glob.glob(os.path.join(args.output, "*_tracked.tif")))
            print(f"Found {len(tracked_files)} tracked files for stacking.")
            tracked = [imread(g) for g in tracked_files]
            # print(f"Number of gradients loaded: {len(tracked)}, with total shape: {tracked[0].shape}")
            
        


    print(f"Stack shape: {stack.shape}, dim: {stack.ndim}")
    # Sauvegarde
    save_path_raw = os.path.join(stack_dir, "raw_image_stacked.tif")
    imwrite(save_path_raw, stack, imagej=True)
    print(f"Cropped images saved to {args.output}.")
           
    launch_3dannotation_viewer(args)

    


def launch_annotator2d(args):
    """
    Function to set the mode to 3D annotator in micro-sam.
    """
    if args.segment:
        print("Segmenting...")
        print(f"Input folder for segmentation: {args.output}")

        print("Segmenting images from output directory...")
        run_cellpose_cli(input_folder=args.input, model_type=args.model, custom_model=args.custom_model, diameter=args.diameter)
        extract_grads_gray(input_folder=args.input)

    launch_2dannotation_viewer(args)
    print("Setting mode to 3D...")  # Placeholder for actual implementation

import torch
def main():

    import torchvision
    print(torchvision.__version__)

    if torch.cuda.is_available():
        print("torch")
    else:
        print("cpu")
    parser = argparse.ArgumentParser(description="Main pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--annotator2d', action='store_true', help="Activate 2D annotator")
    parser.add_argument('--annotator3d', action='store_true', help="Activate 3D annotator")

    parser.add_argument('--crop',  action='store_true', help="Activate cropping")
    parser.add_argument('--input', type=str, required=True, help="Input directory")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--pattern', nargs='+', type=str, help="Pattern to match in filenames")
    parser.add_argument('--n_files', type=int, default=10, help="Number of files to sample")
    parser.add_argument('--crop_size', type=int, default=400, help="Crop size")
    parser.add_argument('--extension', type=str, default=".tif", help="File extension to match")
    parser.add_argument('--stack',  action='store_true', help="Stack images")
    parser.add_argument('--tracking', action='store_true', help="Performs tracking of centroids after segmentation to match labels across frames")

    parser.add_argument('--segment',  action='store_true', help="Activate segmentation")
    parser.add_argument('--diameter', type=int, default=80, help="Diameter of the cells")
    parser.add_argument('--model', type=str, default="cpsam", help="File extension to match")
    parser.add_argument('--custom_model', type=str, default=None, help="Path to custom model for segmentation")
    
    args = parser.parse_args()
    
    if args.annotator2d:
        print("Launching 2D annotator...")
        launch_annotator2d(args)

    elif args.annotator3d:
        print("Launching 3D annotator...")
        launch_annotator3d(args)



if __name__ == "__main__":
    print("main")
    main()
    # print("launching annotation viewer")
    # launch_annotation_viewer()
    #print("split train test")
    #split_dataset(finetune_dir="D:\\micro_sam\\Datasets\\Output", train_ratio=0.8)
    # print("finetuning cellpose")
    # finetune_cellpose()
