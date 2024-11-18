import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    Apply a colormap to a grayscale image (numpy array).
    """
    value = value.copy()  # Copy of input array
    if vmin is None:
        vmin = np.min(value)  # Min value
    if vmax is None:
        vmax = np.max(value)  # Max value
    if cmap is None:
        cmap = "magma"  # Colour used to visualize dmap

    # Normalize value to 0-1
    value = (value - vmin) / (vmax - vmin + 1e-8)
    value = np.clip(value, 0, 1)

    # Apply colormap
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (H, W, 4)

    img = value[:, :, :3]  # Remove alpha channel to get RGB
    return img  # Return RGB img as Numpy array


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(
        os.path.join(INPUT_DIR, "*.jpg")
    )
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading {image_path}")
                continue
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            FINAL_HEIGHT = original_height
            FINAL_WIDTH = original_width

            # Use the infer_image method
            pred = model.infer_image(image)

            predm = pred.squeeze()

            print("Saving images ...")
            # Resize color image and depth to final size
            color_image = Image.fromarray(image)
            resized_color_image = color_image.resize(
                (FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS
            )
            resized_pred = Image.fromarray(predm).resize(
                (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
            )

            focal_length_x, focal_length_y = (
                (FX, FY) if not NYU_DATA else (FL, FL)
            )  # If NYU_DATA is False use FX, FY otherwise FL
            x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
            x = (x - FINAL_WIDTH / 2) / focal_length_x
            y = (y - FINAL_HEIGHT / 2) / focal_length_y
            z = np.array(resized_pred)
            points = np.stack(
                (np.multiply(x, z), np.multiply(y, z), z), axis=-1
            ).reshape(-1, 3)

            # Image.fromarray(predm).convert("L").resize(
            #     (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
            # ).save(
            #     os.path.join(
            #         OUTPUT_DIR,
            #         os.path.splitext(os.path.basename(image_path))[0] + "_pred01.png",
            #     )
            # )

            # p = colorize(predm, cmap="magma_r")
            # Image.fromarray(p).save(
            #     os.path.join(
            #         OUTPUT_DIR,
            #         os.path.splitext(os.path.basename(image_path))[0] + "_pred02.png",
            #     )
            # )

            # pm = colorize(z, cmap="magma_r")
            # Image.fromarray(pm).save(
            #     os.path.join(
            #         OUTPUT_DIR,
            #         os.path.splitext(os.path.basename(image_path))[0] + "_pred03.png",
            #     )
            # )

            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
            imgdepth = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
            o3d.io.write_image(
                os.path.join(
                    OUTPUT_DIR,
                    os.path.splitext(os.path.basename(image_path))[0] + "_pred04.png",
                ),
                imgdepth,
            )

            print(f"Depth range: {z.min()} to {z.max()}")

            # Uncomment the following lines if you want to save point clouds
            # colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         OUTPUT_DIR,
            #         os.path.splitext(os.path.basename(image_path))[0] + ".ply",
            #     ),
            #     pcd,
            # )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process images to generate depth maps and point clouds."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the pre-trained model weights.",
    )
    parser.add_argument(
        "--encoder",
        default="vitl",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Model encoder to use.",
    )
    parser.add_argument(
        "--max-depth",
        default=20,
        type=float,
        choices=[20, 80],  # 20 for indoor model, 80 for outdoor model
        help="Maximum depth value for the depth map.",
    )
    parser.add_argument(
        "--fx", default=1109, type=float, help="Focal length along the x-axis."
    )
    parser.add_argument(
        "--fy", default=1109, type=float, help="Focal length along the y-axis."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hypersim",
        choices=["hypersim", "vkitti"],  # Depends on encoder used (indoor or outdoor)
        help="Dataset type if needed (e.g., nyu).",
    )
    parser.add_argument(
        "--nyu-data", action="store_true", help="Flag to indicate if using NYU dataset."
    )
    args = parser.parse_args()

    # Set the default output directory based on dataset and encoder
    if args.output_dir is None:
        args.output_dir = os.path.join("./output", args.dataset, args.encoder)

    # Set global variables used in process_images
    global OUTPUT_DIR, INPUT_DIR, FX, FY, FL, NYU_DATA, DATASET

    OUTPUT_DIR = args.output_dir
    INPUT_DIR = args.input_dir
    FX = args.fx
    FY = args.fy
    FL = 715.0873  # Focal length - NOT IN USE
    NYU_DATA = args.nyu_data
    DATASET = args.dataset

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Model configuration based on the chosen encoder
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    # Initialize the model
    from depth_anything_v2.dpt import DepthAnythingV2

    depth_anything = DepthAnythingV2(
        **{**model_configs[args.encoder], "max_depth": args.max_depth}
    )
    depth_anything.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Call the process_images function
    process_images(depth_anything)


if __name__ == "__main__":
    main()

# Example usage:
# Will make default output dir if specific one is not provided

# Indoor
# python3 test4.py --input-dir ../data/test/test-input/ --model-path checkpoints/indoor/depth_anything_v2_metric_hypersim_vitl.pth

# Outdoor
# python3 test5.py --input-dir ../data/actual-data/ --model-path checkpoints/outdoor/depth_anything_v2_metric_vkitti_vitl.pth --encoder vitl --max-depth 80 --dataset vkitti
