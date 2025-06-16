import argparse
from Source.segmentation import run_cellpose_cli, tracking_centroids
from Source.crops import generate_random_crops, crop_img, group_images_into_stacks
from Source.annotation_plugin import launch_2dannotation_viewer, launch_3dannotation_viewer
from Source.finetune import finetune_cellpose, split_dataset
import numpy as np
import os
import shutil
import glob
from tifffile import imwrite, imread

from cellpose import models

def launch_annotator2d (args):
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
        # # cropped_img = crop_img(rfiles, output_dir=args.output, size=args.crop_size)
        # # stack = np.stack(cropped_img, axis=0)  # Stack the cropped images along a new dimension
        # # # Forcer la forme (height, width, num_images)
        # # # stack = np.transpose(stack, (1, 2, 0))
        # # # print("After permute:", stack.shape)
        # # print(f"Stack shape: {stack.shape}, dim: {stack.ndim}")
        # # imwrite(os.path.join(args.output, "raw_image_stacked.tif"), stack, imagej=True)  # imagej to write a multi-page TIF compatible with ImageJ, with correctly interpreted dimensions and metadata
        # # print(f"Cropped images saved to {args.output}.")
        
        
    if args.segment:
        print("Segmenting...")
        print(f"Input folder for segmentation: {args.output}")

        if args.crop == False:
            print("Copying original images from input directory to output directory...")
            # Copier tous les fichiers depuis args.input vers args.output
            for filename in os.listdir(args.input):
                src_path = os.path.join(args.input, filename)
                dst_path = os.path.join(args.output, filename)
                if os.path.isfile(src_path):  # Ignore les sous-dossiers
                    shutil.copy(src_path, dst_path)

        print("Segmenting images from output directory...")
        run_cellpose_cli(input_folder=args.output, model_type=args.model, diameter=args.diameter)

        mask_files = sorted(glob.glob(os.path.join(args.output, "*_cp_masks.TIF")))
        print(f"Found {len(mask_files)} mask files for stacking.")
        masks = [imread(m) for m in mask_files]
        print(f"Number of masks loaded: {len(masks)}")

        # # imwrite(os.path.join(args.output, "masks_stacked.tif"), np.stack(masks, axis=0), imagej=True)  # (N, H, W)
        # Sauvegarde
        save_path_mask = os.path.join(stack_dir, "masks_stacked.tif")
        imwrite(save_path_mask, np.stack(masks, axis=0), imagej=True)

        # print("stack shape b4 cellpose", stack.shape)
        # model = models.Cellpose(model_type='cyto')
        # masks, flows, styles, diams = model.eval(stack, diameter=30, channels=[0, 0])
        # print("Masks shape:", masks.shape)

        # for i, mask in enumerate(masks):
        #     imwrite(os.path.join(args.output, f"output_mask{i}_testmathieu.tif"), mask, imagej=True)

    
    # Forcer la forme (height, width, num_images)
    # stack = np.transpose(stack, (1, 2, 0))
    # print("After permute:", stack.shape)
    print(f"Stack shape: {stack.shape}, dim: {stack.ndim}")
    # # imwrite(os.path.join(args.output, "raw_image_stacked.tif"), stack, imagej=True)  # imagej to write a multi-page TIF compatible with ImageJ, with correctly interpreted dimensions and metadata
    # Sauvegarde
    save_path_raw = os.path.join(stack_dir, "raw_image_stacked.tif")
    imwrite(save_path_raw, stack, imagej=True)
    print(f"Cropped images saved to {args.output}.")
           
    launch_3dannotation_viewer(args)

    


def launch_annotator3d ():
    """
    Function to set the mode to 3D annotator in micro-sam.
    """
    launch_3dannotation_viewer()
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

    parser.add_argument('--segment',  action='store_true', help="Activate segmentation")
    parser.add_argument('--diameter', type=int, default=80, help="Diameter of the cells")
    parser.add_argument('--model', type=str, default="cyto3", help="File extension to match")
    
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
