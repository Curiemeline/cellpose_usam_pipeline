import os, random, glob, cv2, tifffile, argparse, re
import numpy as np
from tifffile import imwrite, imread, TiffWriter
from natsort import natsorted

def generate_random_crops(input, n_files, patterns, extension):

    # Get a list of n_files random files in the input directory
    print(input)
    #random_files=random.sample(glob.glob(os.path.join(input, "*.tif")), n_files)
    #random_files=random.sample(glob.glob(os.path.join(input, f"*{pattern}*{extension}")), n_files)

    if patterns is None or len(patterns) == 0:
        patterns = [""]
    print(patterns)
    random_files = random.sample(
        [f for f in natsorted(glob.glob(os.path.join(input, f"*{extension}")))
        if all(p in os.path.basename(f) for p in patterns)],
        n_files
    )
    for f in natsorted(random_files):
        print("sorted random files: ",f)
    return natsorted(random_files)
            


def crop_img(rfiles, output_dir, size):
    list_cropped_img = []  # List to store cropped images
    print(len(rfiles))


    for file in natsorted(rfiles):
        print("cropped: ",file)
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
