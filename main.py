import argparse
from Source.segmentation import run_cellpose_cli, tracking_centroids
from Source.crops import generate_random_crops, crop_img, group_images_into_stacks
from Source.annotation_plugin import launch_2dannotation_viewer, launch_3dannotation_viewer
from Source.finetune import finetune_cellpose
import numpy as np
import os
from tifffile import imwrite
def launch_annotator2d (args):
    """
    Function to set the mode to 2D annotator in micro-sam.
    """
    if args.crop:
        print("Cropping images before launching 2D annotator...")
        rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
        print(f"Generated {len(rfiles)} random crops from the input directory.")
        cropped_img = crop_img(rfiles, output_dir=args.output, size=args.crop_size)
        stack = np.stack(cropped_img, axis=0)  # Stack the cropped images along a new dimension
        imwrite(os.path.join(args.output, "output_stack.tif"), stack)
        print(f"Cropped images saved to {args.output}.")
        
    if args.segment:
        print("Segmenting...")
        run_cellpose_cli(input_folder=args.output, model_type=args.model, diameter=args.diameter)
    
    launch_2dannotation_viewer()

    


def launch_annotator3d ():
    """
    Function to set the mode to 3D annotator in micro-sam.
    """
    launch_3dannotation_viewer()
    print("Setting mode to 3D...")  # Placeholder for actual implementation


def main():
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
    # print("main")
    # main()
    #print("launching annotation viewer")
    # launch_annotation_viewer()
    # print("finetuning cellpose")
    finetune_cellpose()