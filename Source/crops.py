import os, random, glob, cv2, tifffile, argparse, re
import numpy as np
from tifffile import imwrite, imread, TiffWriter
from natsort import natsorted

############################################################################################## WIP 

def group_images_into_stacks(input_folder, output_folder):
    """
    Groups images into stacks and saves them as .tif files.

    Parameters:
    - input_folder: Directory with input images.
    - output_folder: Directory to save stacks.
    """
    
    if not isinstance(input_folder, str):
        raise TypeError(f"Input must be a string representing the file path but got {type(input_folder)}")
    # Regex to parse metadata from filenames
    pattern = r"([^_]+)_w(.+)_s(\d+)_t(\d+).TIF"      # (\d+) is a group that matches any digit. The parentheses are used to group the digits   #TODO Changer le pattern !!!!!
    grouped_files = {}                                                  # Final dictionary that will hold the group of files per image

    # Organize files into groups
    for f in natsorted(os.listdir(input_folder)):                       # For each file from the sorted list of files in the input folder
        if f.startswith("._") or not f.lower().endswith(".tif"):    
            continue                                                    # Skip possible hidden files (start with ._) and unwanted files (not .TIF or .tif) 
        
        match = re.match(pattern, f)                                    # Apply our defined pattern to each file, and only work with the matches

        if match:

            basename, wavelength, stage, time = match.groups()               # match.groups() returns a tuple with the matched groups which are in () in the pattern defined above. So all (\d+). We never use time actually.
            key = (basename, wavelength, stage)                              # This is the unique key, or tuple, for an image

            if key not in grouped_files:                                # If the key is not in the dictionary, meaning we did not group the files for this image yet, create an empty list
                grouped_files[key] = []
                
            grouped_files[key].append(os.path.join(input_folder, f))    # Append at this key, so for this stack of images, the file path of the file we are currently working with
            
    stack_files = []
    # Now that we regrouped every files to its corresponding image, create stacks for each group
    for key, file_list in grouped_files.items():                        # For each key (image) and its corresponding list of files (so all the timeframes 1, 2, 3, ...)

        stack = [imread(file) for file in file_list]                    # For each file in this list, read the image to transform it into numpy array and return a list of numpy array of all your timeframes

        stack_array = np.stack(stack, axis=0)                           # Create a 3D array from the list of numpy array we created before 
        basename, wavelength, stage = key                                    # Retrieve each component of our tuple, key, to name the output file
        output_path = os.path.join(output_folder, f"{basename}_w{wavelength}_s{stage}_stack.tif")
        imwrite(output_path, stack_array)
        stack_files.append(output_path)
        print(f"Saved stack: {output_path}")
        

    return stack_files



# # def generate_random_crops_from_stacks(stack_files, n_crops, pattern, crop_size, output_dir):
# #     """Generate random crops from multiple multi-frame TIFFs."""
# #     crops_created = 0  # Compteur de crops créés

# #     while crops_created < n_crops:
# #         # Choisir un stack au hasard
# #         stack_file = random.choice(stack_files)

# #         # Appliquer le filtre de pattern si spécifié
# #         if pattern and not any(p in os.path.basename(stack_file) for p in pattern):
# #             print(f"Skipping {stack_file}. Does not match the pattern.")
# #             continue  # Skip files that don't match the pattern

# #         stack = tifffile.imread(stack_file)

# #         # Choisir une frame aléatoire dans ce stack
# #         if stack.ndim == 3:
# #             n_frames = stack.shape[0]
# #             frame_idx = random.randint(0, n_frames - 1)
# #             img = stack[frame_idx]
# #         else:
# #             img = stack  # stack avec 1 seule frame

# #         h, w = img.shape[:2]
# #         if h < crop_size or w < crop_size:
# #             print(f"Warning: skipping small frame in {stack_file}")
# #             continue

# #         x = random.randint(0, w - crop_size)
# #         y = random.randint(0, h - crop_size)
# #         crop = img[y:y+crop_size, x:x+crop_size]

# #         # Construire un nom explicite
# #         base = os.path.splitext(os.path.basename(stack_file))[0]
# #         output_path = os.path.join(output_dir, f"{base}_crop_{crops_created:04d}.tif")
# #         tifffile.imwrite(output_path, crop)

# #         crops_created += 1  # Incrémente le compteur

# #         # Affiche une mise à jour si nécessaire
# #         print(f"Crop {crops_created}/{n_crops} created from {stack_file}.")


##############################################################################################

def generate_random_crops(input, n_files, patterns, extension):

    # Get a list of n_files random files in the input directory
    print(input)
    #random_files=random.sample(glob.glob(os.path.join(input, "*.tif")), n_files)
    #random_files=random.sample(glob.glob(os.path.join(input, f"*{pattern}*{extension}")), n_files)

    if patterns is None or len(patterns) == 0:
        patterns = [""]

    random_files = random.sample(
        [f for f in sorted(glob.glob(os.path.join(input, f"*{extension}")))
        if all(p in os.path.basename(f) for p in patterns)],
        n_files
    )
    
    return random_files
            


def crop_img(rfiles, output_dir, size):
    list_cropped_img = []  # List to store cropped images
    print(len(rfiles))

    
    for file in rfiles:
        print(file)
        img = tifffile.imread(file)
        h, w = img.shape[:2]

        if h < size or w < size:
            print(f"Warning: Image {file} is too small ({h}x{w}) and was skipped.")
            continue

        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        cropped_img = img[y:y + size, x:x + size]
        list_cropped_img.append(cropped_img)
        tifffile.imwrite(os.path.join(output_dir, os.path.basename(file)), cropped_img)

    print(len(list_cropped_img))
    return list_cropped_img


if __name__ == "__main__":
    
    INPUT = r"D:\COPPEY\Biolectricity\Dataset\20241105_densities_ibidi_rpe1_mdck"
    OUTPUT = r"D:\micro_sam\Datasets\Output"
    
    rfiles = generate_random_crops(INPUT, 10, ["488"])        # TODO: make it a user input ? using args

    crop_img(rfiles, OUTPUT, size=400)
