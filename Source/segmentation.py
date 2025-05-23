import cv2
import numpy as np
import os
import re
import subprocess

from natsort import natsorted
from skimage.measure import regionprops
from tifffile import imwrite, imread, TiffWriter


## Global variable declaration. To be modified by user.

INPUT_FOLDER = r"D:\micro_sam\Data"
OUTPUT_SUFFIX = "_seg"
MODEL_TYPE = "cyto3"  # Options: 'cyto', 'nuclei', or custom
CHANNELS = [1, 0]  # First is cytoplasm, second is nuclei (0 = grayscale)
DIAMETER = 120  # Approximate diameter of cells (set to 0 for auto-detection)
#OUTPUT_PATH = "D:\\COPPEY\\Biolectricity\\Dataset\\test"


######################################################################## Create stack for each image ########################################################################

def group_images_into_stacks(input_folder, output_folder):
    """
    Groups images into stacks and saves them as .tif files.

    Parameters:
    - input_folder: Directory with input images.
    - output_folder: Directory to save stacks.
    """

    # Regex to parse metadata from filenames
    pattern = r"(first|second|third)-day_w(\d+)_s(\d+)_t(\d+).TIF"      # (\d+) is a group that matches any digit. The parentheses are used to group the digits
    grouped_files = {}                                                  # Final dictionary that will hold the group of files per image

    # Organize files into groups
    for f in natsorted(os.listdir(input_folder)):                       # For each file from the sorted list of files in the input folder
        if f.startswith("._") or not f.lower().endswith(".tif"):    
            continue                                                    # Skip possible hidden files (start with ._) and unwanted files (not .TIF or .tif) 
        
        match = re.match(pattern, f)                                    # Apply our defined pattern to each file, and only work with the matches

        if match:
            day, wavelength, stage, time = match.groups()               # match.groups() returns a tuple with the matched groups which are in () in the pattern defined above. So all (\d+). We never use time actually.
            key = (day, wavelength, stage)                              # This is the unique key, or tuple, for an image

            if key not in grouped_files:                                # If the key is not in the dictionary, meaning we did not group the files for this image yet, create an empty list
                grouped_files[key] = []
                
            grouped_files[key].append(os.path.join(input_folder, f))    # Append at this key, so for this stack of images, the file path of the file we are currently working with
            

    # Now that we regrouped every files to its corresponding image, create stacks for each group
    for key, file_list in grouped_files.items():                        # For each key (image) and its corresponding list of files (so all the timeframes 1, 2, 3, ...)

        stack = [imread(file) for file in file_list]                    # For each file in this list, read the image to transform it into numpy array and return a list of numpy array of all your timeframes

        stack_array = np.stack(stack, axis=0)                           # Create a 3D array from the list of numpy array we created before 
        day, wavelength, stage = key                                    # Retrieve each component of our tuple, key, to name the output file
        output_path = os.path.join(output_folder, f"{day}-day_w{wavelength}_s{stage}_stack.tif")
        imwrite(output_path, stack_array)
        print(f"Saved stack: {output_path}")
        

    return stack_array


######################################################################## Cellpose segmentation ########################################################################

def run_cellpose_cli(input_folder, model_type, diameter, chan1=1, chan2=0):

    
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
        "--verbose",
        "--dir", input_folder,                  # Directory containing the images
        "--pretrained_model", model_type,       # Model type
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
    






######################################################################## NEW VERSION OF CELLPOSE_CLI WITH FILTERED IMAGES ########################################################################



import os
import shutil
import subprocess
import tempfile

# # def run_cellpose_cli(input_folder, model_type, diameter, patterns=None, chan1=1, chan2=0, output_folder=None):
# #     """
# #     Runs Cellpose CLI for segmentation on selected images and saves the mask as .npy.

# #     Parameters:
# #     - input_folder (str): Path to the input image folder.
# #     - model_type (str): Pretrained model type (e.g., "cyto", "nuclei").
# #     - diameter (int): Estimated cell diameter.
# #     - patterns (list of str, optional): List of substrings to filter filenames.
# #     - chan1 (int): Channel for grayscale.
# #     - chan2 (int): Secondary channel for grayscale.
# #     - output_folder (str, optional): Directory to save segmentation outputs.

# #     Returns:
# #     - list: Paths to the generated mask files.
# #     - str: Any stderr output for debugging.
# #     """

# #     # Vérifier si un filtrage est nécessaire
# #     if patterns:
# #         # Filtrer les fichiers contenant un des motifs
# #         filtered_files = [f for f in os.listdir(input_folder) if any(p in f for p in patterns)]

# #         if not filtered_files:
# #             print("Aucun fichier correspondant aux critères trouvés.")
# #             return None, "No matching files found."

# #         # Créer un dossier temporaire
# #         with tempfile.TemporaryDirectory() as temp_dir:
# #             for file in filtered_files:
# #                 shutil.copy(os.path.join(input_folder, file), temp_dir)
            
# #             selected_folder = temp_dir  # Cellpose va exécuter dans ce dossier
            
# #             # Exécuter Cellpose
# #             command = [
# #                 "cellpose",
# #                 "--verbose",
# #                 "--dir", selected_folder,         
# #                 "--pretrained_model", model_type,
# #                 "--chan", str(chan1),             
# #                 "--chan2", str(chan2),            
# #                 "--diameter", str(diameter),      
# #                 "--save_tif"
# #             ]
# #             print("Executing command:", " ".join(command))
# #             result = subprocess.run(command, capture_output=True, text=True)

# #             if result.returncode != 0:
# #                 print("Cellpose CLI failed to run.")
# #                 print("STDERR:", result.stderr)
# #                 return None, result.stderr
            
# #             # Si un dossier de sortie est spécifié, copier les résultats dedans
# #             if output_folder:
# #                 os.makedirs(output_folder, exist_ok=True)
# #                 for file in os.listdir(selected_folder):
# #                     if file.endswith(".tif") or file.endswith(".npy"):  # Garde uniquement les outputs
# #                         shutil.move(os.path.join(selected_folder, file), os.path.join(output_folder, file))

# #             return [os.path.join(output_folder, f) for f in os.listdir(output_folder)], None
# #     else:
# #         # Aucun filtrage -> utiliser directement le dossier d'origine
# #         selected_folder = input_folder

# #         # Exécuter Cellpose
# #         command = [
# #             "cellpose",
# #             "--verbose",
# #             "--dir", selected_folder,
# #             "--pretrained_model", model_type,
# #             "--chan", str(chan1),
# #             "--chan2", str(chan2),
# #             "--diameter", str(diameter),
# #             "--save_tif"
# #         ]
# #         print("Executing command:", " ".join(command))
# #         result = subprocess.run(command, capture_output=True, text=True)

# #         if result.returncode != 0:
# #             print("Cellpose CLI failed to run.")
# #             print("STDERR:", result.stderr)
# #             return None, result.stderr

# #         return [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.endswith(".tif") or f.endswith(".npy")], None


















######################################################################## Tracking using Centroids ########################################################################

def tracking_centroids(input_folder):
    
    for filename in natsorted(os.listdir(input_folder)):                                    # Loop over now segmented files
        #if filename.endswith("first-day_w1445_s4_stack_seg.npy"):                          # Process only one mask file to test the code
        if filename.endswith("seg.npy"):                                                    # Process only mask files

            mask_path = os.path.join(input_folder, filename)                                # Get path to mask file
            data = np.load(mask_path, allow_pickle=True).item()                             # Load raw image stack because the .npy file doesn't only contain mask but also outlines, probabilities, flows, etc.
            masks = data['masks']                                                           # and retrieve only the mask
            
            relabeled_masks = np.zeros_like(masks)                                          # Initialize an empty array that has the same shape as original mask to store the relabeled masks
            # # tracked_labels = {}                                                             # Dictionary to track labels across frames
            
            for t in range(masks.shape[0]):                                                 # Process each timeframe. Here masks.shape[0] = 6 so it's gonna loop through the mask at timeframes 0, 1, 2, 3, 4, 5
                frame_mask = masks[t]
                
                # # props = regionprops_table(                                                  # Using the library skimage.measure.regionprops_table,
                # #     frame_mask,                                                             # Compute for the current timeframe,
                # #     properties=['label', 'centroid']                                        # each cells label and centroid automatically, and store them in a dictionary called props
                # # )
                
                # # frame_props = pd.DataFrame(props)                                           # Convert Dictionary to DataFrame for easier handling
                # # frame_props['Time'] = t                                                     # Add timepoint column
                
                # # for i, row in frame_props.iterrows():                                       # Loop over each row in the DataFrame and retrieve centroids and labels
                # #     centroid_y, centroid_x = row['centroid-0'], row['centroid-1']
                # #     label_id = row['label']
                    #print(frame_mask[int(centroid_y), int(centroid_x)])

                if t == 0:                                                              # First frame: Initialize tracking
                    relabeled_masks[t] = frame_mask                                     # Copy original labels

                    # # for label_id in np.unique(frame_mask):                              # Loop over all unique labels in the original mask  
                    # #     if label_id == 0:                                               # Skip background labels
                    # #         continue
                    # #     tracked_labels[label_id] = [(t, label_id)]  

                else:                                                                   # If not the first frame
                                                                                        # Check if the centroid lies in a label from the previous frame
                    prev_frame_mask = relabeled_masks[t - 1]
                    current_frame_relabeled = np.zeros_like(frame_mask)                 # Create a new mask for next frame

                    # Compute centroids for all labels in the previous frame
                    prev_props = regionprops(prev_frame_mask)               # Retrieve all labels from previous mask
                    prev_centroids = {int(region.label): tuple(map(int, region.centroid)) for region in prev_props}


                    # # for prev_region in prev_props:                          # For all the previous labels
                    # #     prev_label = prev_region.label                      # Retrieve ID label
                    # #     centroid_y, centroid_x = map(int, prev_region.centroid)  # Retrieve previous labels' centroids and convert them to int 

                    # #     # Check if the previous centroids fall inside a label in the current frame
                    # #     current_label = frame_mask[centroid_y, centroid_x]  # Stores the number of the label where the centroids fall

                    # #     if current_label > 0:  # If the centroid is inside a label
                    # #         # Propagate the previous label to the current frame
                    # #         current_frame_relabeled[frame_mask == current_label] = prev_label   # All the pixels where the actual region we're looking at, so meaning where we have pixels = current label, for example = 3, we reassign all this region with the label of the previous frame
                    # # # Update the relabeled masks for the current frame
                    # # relabeled_masks[t] = current_frame_relabeled            # DANS LE NOUVEAU MASK on met la region réassignée
                    for prev_label, (centroid_y, centroid_x) in prev_centroids.items():
                        if 0 <= centroid_y < frame_mask.shape[0] and 0 <= centroid_x < frame_mask.shape[1]:
                            current_label = frame_mask[centroid_y, centroid_x]
                            if current_label > 0:  # Si le centroid tombe bien sur une cellule
                                ##tracked_labels[current_label] = prev_label  # Assigner le label précédent
                                current_frame_relabeled[frame_mask == current_label] = prev_label

                    relabeled_masks[t] = current_frame_relabeled
            output_path = os.path.join(input_folder, filename.replace("_seg.npy", "_tracked.tif"))
            imwrite(output_path, relabeled_masks)



######################################################################## Contours ########################################################################







#run_cellpose_cli(INPUT_FOLDER, model_type=MODEL_TYPE, diameter=DIAMETER)
