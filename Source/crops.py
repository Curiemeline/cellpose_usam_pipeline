import os, random, glob, cv2, tifffile
import numpy as np

INPUT = r"D:\COPPEY\Biolectricity\Dataset\20241105_densities_ibidi_rpe1_mdck"
OUTPUT = r"D:\micro_sam\Data"

def generate_random_crops(input, n_files, patterns, extension=".tif"):

    # Get a list of n_files random files in the input directory
    print(input)
    #random_files=random.sample(glob.glob(os.path.join(input, "*.tif")), n_files)
    #random_files=random.sample(glob.glob(os.path.join(input, f"*{pattern}*{extension}")), n_files)


    random_files = random.sample(
        [f for f in glob.glob(os.path.join(input, f"*{extension}"))
        if all(p in os.path.basename(f) for p in patterns)],
        n_files
    )
    
    return random_files
            


def crop_img(rfiles, output_dir, size):
    for file in rfiles:
        print(file)
        img = tifffile.imread(file)
        h, w = img.shape[:2]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        crop_img = img[y:y + size, x:x + size]
        tifffile.imwrite(os.path.join(output_dir, os.path.basename(file)), crop_img)


        
rfiles = generate_random_crops(INPUT, 10, ["488"])        # TODO: make it a user input ? using args

crop_img(rfiles, OUTPUT, size=400)
