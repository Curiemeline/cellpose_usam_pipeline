import argparse
from Source.crops import generate_random_crops, crop_img

def main():
    parser = argparse.ArgumentParser(description="Main pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--crop', required=True, action='store_true', help="Activate cropping")
    parser.add_argument('--input', type=str, required=True, help="Input directory")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    parser.add_argument('--pattern', nargs='+', type=str, help="Pattern to match in filenames")
    parser.add_argument('--n_files', type=int, default=10, help="Number of files to sample")
    parser.add_argument('--crop_size', type=int, default=400, help="Crop size")
    parser.add_argument('--extension', type=str, default=".tif", help="File extension to match")
    args = parser.parse_args()

    if args.crop:
        print("Cropping images...")
        # Generate random crops
        rfiles = generate_random_crops(input=args.input, n_files=args.n_files, patterns=args.pattern, extension=args.extension)
        
        # Crop images and save to the output directory
        crop_img(rfiles, output_dir=args.output, size=args.crop_size)
    else:
        # Handle the case where --crop is not provided
        print("No cropping operation performed.")

if __name__ == "__main__":
    main()