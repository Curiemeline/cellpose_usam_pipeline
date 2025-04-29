import argparse
from Source.segmentation import run_cellpose_cli, tracking_centroids

def main():
    parser = argparse.ArgumentParser(description="Main pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    
    if args.segment:
        print("Segmentation is activated.")
        run_cellpose_cli(input_folder=args.output, model_type=args.model, diameter=args.diameter)
        #tracking_centroids(input_folder=args.output)    # TODO Check si c'est des images 2D Pas besoin de tracking
    else:
        # Handle the case where neither --stack nor --crop is provided
        print("No stack or crop operation performed.")


if __name__ == "__main__":
    main()