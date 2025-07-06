import os, random, glob, tifffile

def generate_random_crops(input, n_files, patterns, extension):

    # Get a list of n_files random files in the input directory

    # if patterns is None or len(patterns) == 0:
    #     patterns = [""]
    patterns = [""] if patterns is None or len(patterns) == 0 else patterns
    
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
