import numpy as np
import os
import subprocess

from natsort import natsorted
from skimage.measure import regionprops
from tifffile import imwrite, imread

## Global variable declaration. To be modified by user.

INPUT_FOLDER = r"D:\micro_sam\Datasets\Output"
OUTPUT_SUFFIX = "_seg"
MODEL_TYPE = "cyto3"  # Options: 'cyto', 'nuclei', or custom
CHANNELS = [1, 0]  # First is cytoplasm, second is nuclei (0 = grayscale)
DIAMETER = 120  # Approximate diameter of cells (set to 0 for auto-detection)
#OUTPUT_PATH = "D:\\COPPEY\\Biolectricity\\Dataset\\test"


######################################################################## Cellpose segmentation ########################################################################

def run_cellpose_cli(input_folder, model_type, custom_model, diameter, chan1=1, chan2=0):

    
    """
    Runs Cellpose CLI for segmentation and saves the mask as .npy

    Parameters:
    - input_image (str): Path to the input image file.
    - output_dir (str): Directory where the output masks will be saved.
    - model_type (str): Pretrained model type (e.g., "cyto", "nuclei").
    - diameter (int): Estimated cell diameter.
    - chan1 (int): Channel for grayscale
    - chan2 (int): Secondary channel for grayscale

    Returns:
    - str: Path to the generated mask .npy file.
    - str: Any stderr output for debugging.
    """

    # Construct the CLI command
    command = [
        "cellpose",
        "--use_gpu",                            # Use GPU if available
        "--verbose",
        "--dir", input_folder,                  # Directory containing the images
        "--pretrained_model", model_type,       # Model type
        "--add_model", str(custom_model), 
        "--chan", str(chan1),                   # Channel for grayscale
        "--chan2", str(chan2),                  # Secondary channel
        "--diameter", str(diameter),            # Cell diameter
        "--save_tif"                            # Later, export it on cellpose_napari -> right click on the layer -> Convert to labels
    ]

    # Debugging: Print the command and types
    print("Command List:", command)
    print("Argument Types:", [type(arg) for arg in command])
    
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print("Cellpose CLI failed to run.")
        print("STDERR:", result.stderr)
        return None, result.stderr
    

def extract_grads_gray(input_folder):

    for f in os.listdir(input_folder):

        if f.endswith("seg.npy"):
            mask_path = os.path.join(input_folder, f)
            data = np.load(mask_path, allow_pickle=True).item()
            gradsXY = data['flows'][0]  # Extract cell probabilities
            gradsXY_gray = np.squeeze(np.dot(gradsXY[..., :3], [0.2989, 0.5870, 0.1140]))   # Pour enlever la première dimension, sinon ça donne (1, 400, 400, 3) et c'est pas accepté par microsam pour compute les embeddings. C'est soit 2D (400,400), soit 3D (10, 400, 400)
            imwrite(os.path.join(input_folder, f.replace("seg.npy", "gradsXY_gray.tif")), gradsXY_gray)

if __name__ == "__main__":
    # Example usage
    input_folder = "D:\micro_sam\Datasets\Output"

    extract_grads_gray(input_folder)

######################################################################## Tracking using Centroids ########################################################################

def tracking_centroids(input_folder):
    
    for filename in natsorted(os.listdir(input_folder)):
        if ~filename.startswith("masks"): 
            continue                                  # Loop over now segmented files                       # Process only one mask file to test the code                                                    # Process only mask files

        mask_path = os.path.join(input_folder, filename)                                # Get path to mask file
        # data = np.load(mask_path, allow_pickle=True).item()                             # Load raw image stack because the .npy file doesn't only contain mask but also outlines, probabilities, flows, etc.
        # masks = data['masks']  
        masks = imread(mask_path)                                                         # and retrieve only the mask
        print(f"Processing {filename} with shape {masks.shape}")                              # Print the shape of the mask to check if it is correct
        relabeled_masks = np.zeros_like(masks)                                          # Initialize an empty array that has the same shape as original mask to store the relabeled masks
        
        for t in range(masks.shape[0]):                                                 # Process each timeframe. Here masks.shape[0] = 6 so it's gonna loop through the mask at timeframes 0, 1, 2, 3, 4, 5
            frame_mask = masks[t]

            if t == 0:
                print("t0", t)                                                              # First frame: Initialize tracking
                relabeled_masks[t] = frame_mask
                continue                                     # Copy original labels
                                                                  # If not the first frame
            # Check if the centroid lies in a label from the previous frame
            prev_frame_mask = relabeled_masks[t - 1]
            current_frame_relabeled = np.zeros_like(frame_mask)                 # Create a new mask for next frame

            # Compute centroids for all labels in the previous frame
            prev_props = regionprops(prev_frame_mask)               # Retrieve all labels from previous mask
            prev_centroids = {int(region.label): tuple(map(int, region.centroid)) for region in prev_props}

            for prev_label, (centroid_y, centroid_x) in prev_centroids.items():
                if 0 <= centroid_y < frame_mask.shape[0] and 0 <= centroid_x < frame_mask.shape[1]:
                    current_label = frame_mask[centroid_y, centroid_x]
                    if current_label > 0:  # Si le centroid tombe bien sur une cellule
                        current_frame_relabeled[frame_mask == current_label] = prev_label

            relabeled_masks[t] = current_frame_relabeled

        output_path = os.path.join(input_folder, filename.replace("masks", "tracked"))
        imwrite(output_path, relabeled_masks)
        print(f"Tracking completed for {filename}. Output saved to {output_path}")
