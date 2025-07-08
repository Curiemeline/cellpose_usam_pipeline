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
import cv2
from cellpose import models
from natsort import natsorted

def unstack_images(input_folder, output_folder):
    """
    Unstack images from a folder and save them in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".tif", ".tiff", ".TIF", ".TIFF")):     
            
            filepath = os.path.join(input_folder, filename)
            img = imread(filepath)
            if img.ndim > 2:        # Check if the image has more than 2 dimensions, hence is a stack
                print(f"Unstacking {filename}...")
                for i in range(img.shape[0]):
                    basename = os.path.splitext(os.path.basename(filename))[0]  # os.path.splitext() splits the filename into basename and extension. Returns a tuple where the 1st element is the basename and the 2nd the extension ('.tif' in this example).
                    imwrite(os.path.join(output_folder, f"{basename}_{i}.tif"), img[i], imagej=True)
                    
                    
# # # # def launch_annotator3d(args):
# # # #     """
# # # #     Function to set the mode to 2D annotator in micro-sam.
# # # #     """
# # # #     # Unstack images if they are in a stack format
# # # #     unstack_images(args.input, args.output)  

# # # #     # Créer le chemin vers le dossier Stack (dans le dossier parent)
# # # #     parent_dir = os.path.abspath(os.path.join(args.output, os.pardir))
# # # #     stack_dir = os.path.join(parent_dir, "Stack")
# # # #     os.makedirs(stack_dir, exist_ok=True)

# # # #     if args.crop == False and args.segment == False:
# # # #         print("no cropping nor segmentation, loading images from input directory...")

# # # #         image_files = sorted([
# # # #             os.path.join(args.input, f)
# # # #             for f in os.listdir(args.input)
# # # #             if f.lower().endswith((".tif", ".tiff", ".TIF", ".TIFF"))
# # # #         ])

# # # #         img_list = []
# # # #         for img_path in image_files:
# # # #             print(img_path)
# # # #             img = imread(str(img_path))
# # # #             img_list.append(img)
        
# # # #         stack = np.stack(img_list, axis=0)  # Stack the images along a new dimension
# # # #         print("stacked original images")

# # # #     if args.crop:
# # # #         print("Cropping images before launching 2D annotator...")
# # # #         rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
# # # #         print(f"Generated {len(rfiles)} random crops from the input directory.")
# # # #         cropped_img = crop_img(sorted(rfiles), output_dir=args.output, size=args.crop_size)

# # # #         stack = np.stack(cropped_img, axis=0)  # Stack the cropped images along a new dimension
# # # #         # # cropped_img = crop_img(rfiles, output_dir=args.output, size=args.crop_size)
# # # #         # # stack = np.stack(cropped_img, axis=0)  # Stack the cropped images along a new dimension
# # # #         # # # Forcer la forme (height, width, num_images)
# # # #         # # # stack = np.transpose(stack, (1, 2, 0))
# # # #         # # # print("After permute:", stack.shape)
# # # #         # # print(f"Stack shape: {stack.shape}, dim: {stack.ndim}")
# # # #         # # imwrite(os.path.join(args.output, "raw_image_stacked.tif"), stack, imagej=True)  # imagej to write a multi-page TIF compatible with ImageJ, with correctly interpreted dimensions and metadata
# # # #         # # print(f"Cropped images saved to {args.output}.")
        
        
# # # #     if args.segment:
# # # #         print("Segmenting...")
# # # #         print(f"Input folder for segmentation: {args.output}")

# # # #         if args.crop == False:
# # # #             print("No cropping, copying original images from input directory to output directory...")
# # # #             patterns = args.pattern
# # # #             print("patterns:", patterns)
# # # #             if patterns is None or len(patterns) == 0:
# # # #                 patterns = [""]
# # # #             else:
# # # #                 print("Copying original images from input directory to output directory...")
# # # #                 # Copier tous les fichiers depuis args.input vers args.output
# # # #                 for filename in os.listdir(args.input):
# # # #                     if all(p in filename for p in patterns):
# # # #                         # Vérifier si le fichier correspond au modèle
# # # #                         src_path = os.path.join(args.input, filename)
# # # #                         dst_path = os.path.join(args.output, filename)
# # # #                         if os.path.isfile(src_path):  # Ignore les sous-dossiers
# # # #                             shutil.copy(src_path, dst_path)

# # # #         print("Resizing images in output directory to 512x512...")
# # # #         for filename in os.listdir(args.output):
# # # #             print(filename)
# # # #             frame = imread(os.path.join(args.output, filename)) 
# # # #             frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
# # # #             imwrite(os.path.join(os.basename(args.output), filename), frame, imagej=True)
# # # #         print("Segmenting images from output directory...")
# # # #         run_cellpose_cli(input_folder=args.output, model_type=args.model, custom_model=args.custom_model, diameter=args.diameter)
# # # #         extract_grads_gray(input_folder=args.output)
        

# # # #         mask_files = sorted(glob.glob(os.path.join(args.output, "*_cp_masks.tif")))
# # # #         print(f"Found {len(mask_files)} mask files for stacking.")
# # # #         masks = [imread(m) for m in mask_files]
# # # #         print(f"Number of masks loaded: {len(masks)}, with total shape: {masks[0].shape}")

# # # #         grads_files = sorted(glob.glob(os.path.join(args.output, "*_gradsXY_gray.tif")))
# # # #         print(f"Found {len(grads_files)} gradient files for stacking.")
# # # #         grads = [imread(g) for g in grads_files]
# # # #         print(f"Number of gradients loaded: {len(grads)}, with total shape: {grads[0].shape}")

        

# # # #         # # imwrite(os.path.join(args.output, "masks_stacked.tif"), np.stack(masks, axis=0), imagej=True)  # (N, H, W)
# # # #         # Sauvegarde
# # # #         save_path_mask = os.path.join(stack_dir, "masks_stacked.tif")
# # # #         imwrite(save_path_mask, np.stack(masks, axis=0), imagej=True)

# # # #         save_path_grads = os.path.join(stack_dir, "grads_stacked.tif")
# # # #         imwrite(save_path_grads, np.stack(grads, axis=0))

# # # #         if args.tracking:
# # # #             tracking_centroids(input_folder=stack_dir)
            
# # # #             tracked_files = sorted(glob.glob(os.path.join(args.output, "*_tracked.tif")))
# # # #             print(f"Found {len(tracked_files)} tracked files for stacking.")
# # # #             tracked = [imread(g) for g in tracked_files]
# # # #             #print(f"Number of gradients loaded: {len(tracked)}, with total shape: {tracked[0].shape}")
            
        


    
# # # #     # Forcer la forme (height, width, num_images)
# # # #     # stack = np.transpose(stack, (1, 2, 0))
# # # #     # print("After permute:", stack.shape)
# # # #     print(f"Stack shape: {stack.shape}, dim: {stack.ndim}")
# # # #     # # imwrite(os.path.join(args.output, "raw_image_stacked.tif"), stack, imagej=True)  # imagej to write a multi-page TIF compatible with ImageJ, with correctly interpreted dimensions and metadata
# # # #     # Sauvegarde
# # # #     save_path_raw = os.path.join(stack_dir, "raw_image_stacked.tif")
# # # #     imwrite(save_path_raw, stack, imagej=True)
# # # #     print(f"Cropped images saved to {args.output}.")
           
# # # #     launch_3dannotation_viewer(args)

    



def launch_annotator3d(args):
    """
    Launch the 2D annotator pipeline with optional unstacking, cropping, segmentation, and tracking.
    """

    # Créer le dossier output et Stack
    os.makedirs(args.output, exist_ok=True)
    parent_dir = os.path.abspath(os.path.join(args.output, os.pardir))
    stack_dir = os.path.join(parent_dir, "Stack")
    os.makedirs(stack_dir, exist_ok=True)

    # === 1. UNSTACK LES IMAGES SI NÉCESSAIRE ===
    print("Checking for stacks to unstack...")
    unstack_images(args.input, args.output)

    # === 2. COPIE DES IMAGES SI PAS DE CROP MAIS SEGMENTATION ===
    if not args.crop and args.segment:
        img_list = []
        print("Copying images from input to output for segmentation...")
        patterns = args.pattern if args.pattern else [""]
        for filename in natsorted(os.listdir(args.input)):
            if all(p in filename for p in patterns) and filename.lower().endswith((".tif", ".tiff")):
                src_path = os.path.join(args.input, filename)
                dst_path = os.path.join(args.output, filename)
                # if os.path.isfile(src_path):
                #     shutil.copy(src_path, dst_path)
                if not os.path.isfile(src_path):
                    continue

                try:
                    img = imread(src_path)
                    if img.ndim == 2:
                        shutil.copy(src_path, dst_path)
                        print(f"Copied 2D image: {filename}")
                        img_list.append(imread(dst_path))
                    else:
                        print(f"Skipped stack or invalid: {filename}, ndim={img.ndim}, shape={img.shape}")
                except Exception as e:
                    print(f"Failed to read {filename}: {e}")
            
        if img_list: imwrite(os.path.join(stack_dir, "raw_image_stacked.tif"), np.stack(img_list, axis=0), imagej=True)


    # === 3. GÉNÈRE DES CROPS SI NÉCESSAIRE ===
    if args.crop:
        print("Cropping images before launching 2D annotator...")
        rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
        print(f"Generated {len(rfiles)} random crops from the input directory.")
        cropped_img = crop_img(sorted(rfiles), output_dir=args.output, size=args.crop_size)
        stack = np.stack(cropped_img, axis=0)

        imwrite(os.path.join(stack_dir, "raw_image_stacked.tif"), stack, imagej=True)
        print(f"Saved raw image stack to {stack_dir}/raw_image_stacked.tif")

    # === 4. SEGMENTATION SI DEMANDÉE ===
    if args.segment:
        print("Resizing images in output directory to 512x512 before segmentation...")
        img_list = []
        for filename in natsorted(os.listdir(args.output)):     # natsorted is VERY important because it doesn't read in alphabetical order, so the original film is not read in the right order and messes up overlay with masks.
            if filename.lower().endswith((".tif", ".tiff")):
                frame_path = os.path.join(args.output, filename)
                frame = imread(frame_path)
                resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
                imwrite(frame_path, resized, imagej=True)
                img_list.append(resized)
        stack = np.stack(img_list, axis=0)  # Stack the resized images along a new dimension
        imwrite(os.path.join(stack_dir, "raw_image_stacked.tif"), stack, imagej=True)

        print("Running Cellpose segmentation...")
        run_cellpose_cli(input_folder=args.output, model_type=args.model, custom_model=args.custom_model, diameter=args.diameter)
        extract_grads_gray(input_folder=args.output)

        # Chargement des masques
        mask_files = sorted(glob.glob(os.path.join(args.output, "*_cp_masks.tif")))
        print(f"Found {len(mask_files)} mask files.")
        masks = [imread(m) for m in mask_files]

        grads_files = sorted(glob.glob(os.path.join(args.output, "*_gradsXY_gray.tif")))
        print(f"Found {len(grads_files)} gradient files.")
        grads = [imread(g).astype(np.float32) for g in grads_files]

        imwrite(os.path.join(stack_dir, "masks_stacked.tif"), np.stack(masks, axis=0), imagej=True)
        imwrite(os.path.join(stack_dir, "grads_stacked.tif"), np.stack(grads, axis=0), imagej=True)

        if args.tracking:
            tracking_centroids(input_folder=stack_dir)


    # # === 5. STACK DES IMAGES BRUTES POUR LA VISU 3D ===
    # if not args.crop:
    #     print("Stacking raw images from output folder...")
    #     raw_files = sorted([
    #         os.path.join(args.output, f)
    #         for f in os.listdir(args.output)
    #         if f.lower().endswith((".tif", ".tiff")) and "_cp_masks" not in f and "_gradsXY" not in f and "_tracked" not in f
    #     ])
    #     img_list = [imread(f) for f in raw_files]
    #     stack = np.stack(img_list, axis=0)

        imwrite(os.path.join(stack_dir, "raw_image_stacked.tif"), stack, imagej=True)
        print(f"Saved raw image stack to {stack_dir}/raw_image_stacked.tif")

    # === 6. LANCEMENT DU VIEWER ===
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
    parser.add_argument('--diameter', type=int, default=None, help="Diameter of the cells")
    parser.add_argument('--model', type=str, default="cpsam", help="File extension to match")
    parser.add_argument('--custom_model', type=str, default=None, help="Path to custom model for segmentation")
    
    args = parser.parse_args()
    
    if args.annotator2d:
        print("Launching 2D annotator...")
        launch_annotator2d(args)

    elif args.annotator3d:
        print("Launching 3D annotator...")
        launch_annotator3d(args)

    # # if args.segment:
    # #     print("Segmentation is activated.")
    # #     run_cellpose_cli(input_folder=args.output, model_type=args.model, diameter=args.diameter)
    # #     #tracking_centroids(input_folder=args.output)    # TODO Check si c'est des images 2D Pas besoin de tracking

    # # if args.stack and args.crop:
    # #     print("Stacking images and cropping...")
    # #     stack_files = group_images_into_stacks(input_folder=args.input, output_folder=args.output)
    # #     generate_random_crops_from_stacks(stack_files, n_crops=args.n_files, pattern=args.pattern, crop_size=args.crop_size, output_dir=args.output)
    
    # # elif args.stack:
    # #     print("Stacking images...")
    # #     stack_files = group_images_into_stacks(input_folder=args.input, output_folder=args.output)
    
    # # elif args.crop:
    # #     print("Cropping images...")
    # #     # Generate random crops
    # #     rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
    # #     crop_img(rfiles, output_dir=args.output, size=args.crop_size)
    
    # # else:
    # #     # Handle the case where neither --stack nor --crop is provided
    # #     print("No stack or crop operation performed.")



if __name__ == "__main__":
    print("main")
    main()
    # print("launching annotation viewer")
    # launch_annotation_viewer()
    #print("split train test")
    #split_dataset(finetune_dir="D:\\micro_sam\\Datasets\\Output", train_ratio=0.8)
    # print("finetuning cellpose")
    # finetune_cellpose()
