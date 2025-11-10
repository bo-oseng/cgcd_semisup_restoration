import os
import argparse
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def extract_patches(image, patch_size, stride):
    """Extract patches from a single image with given patch_size and stride."""
    patches = []
    w, h = image.size

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_folder(input_folder, output_folder, patch_size, stride, max_patches_per_image):
    """Process all images in input_folder and save random patches to output_folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"}

    # Get all image files
    image_files = [f for f in input_path.iterdir() if f.suffix in image_extensions]

    print(f"Processing {len(image_files)} images from {input_folder}")
    print(f"Max patches per image: {max_patches_per_image}")

    patch_count = 0
    for img_file in tqdm(image_files, desc=f"Processing {input_path.name}"):
        try:
            image = Image.open(img_file).convert("RGB")
            patches = extract_patches(image, patch_size, stride)

            # Randomly select up to max_patches_per_image patches
            selected_patches = random.sample(patches, min(max_patches_per_image, len(patches)))

            # Save selected patches
            img_name = img_file.stem
            for i, patch in enumerate(selected_patches):
                patch_name = f"{img_name}_patch_{i:04d}.png"
                patch_path = output_path / patch_name
                patch.save(patch_path)
                patch_count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print(f"Saved {patch_count} patches to {output_folder}")
    return patch_count


# python preprocess_random_patches.py --input_folder /mnt/sdd/kbs/proxyrestore/SemiAIORCL/data/CDD_11/Train --output_folder /mnt/sdd/kbs/proxyrestore/SemiAIORCL/data/CDD_11/Pseudo --patch_size 256 --stride 200 --max_patches 8 --recursive
def main():
    parser = argparse.ArgumentParser(description="Extract random patches from images for training")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing images")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder to save patches")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches (default: 256)")
    parser.add_argument("--stride", type=int, default=200, help="Stride for patch extraction (default: 200)")
    parser.add_argument("--max_patches", type=int, default=3, help="Maximum patches per image (default: 3)")
    parser.add_argument("--recursive", action="store_true", help="Process subfolders recursively")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    input_path = Path(args.input_folder)

    if not input_path.exists():
        print(f"Error: Input folder {args.input_folder} does not exist")
        return

    if args.recursive:
        # Process subfolders
        subfolders = [f for f in input_path.iterdir() if f.is_dir()]
        if not subfolders:
            # No subfolders, process the folder itself
            process_folder(args.input_folder, args.output_folder, args.patch_size, args.stride, args.max_patches)
        else:
            # Process each subfolder
            for subfolder in subfolders:
                relative_path = subfolder.relative_to(input_path)
                output_subfolder = Path(args.output_folder) / relative_path
                process_folder(str(subfolder), str(output_subfolder), args.patch_size, args.stride, args.max_patches)
    else:
        # Process single folder
        process_folder(args.input_folder, args.output_folder, args.patch_size, args.stride, args.max_patches)


if __name__ == "__main__":
    main()
