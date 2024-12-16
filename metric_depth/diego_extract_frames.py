import os
import subprocess
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos at a given frame rate."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second to extract."
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Video file extension to look for (default: mp4).",
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    output_dir = args.output_dir
    fps = args.fps
    extension = args.ext

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all videos in video_dir with the given extension
    videos = [
        f for f in os.listdir(video_dir) if f.lower().endswith("." + extension.lower())
    ]
    if not videos:
        print(f"No .{extension} files found in {video_dir}")
        sys.exit(1)

    for video in videos:
        base_name = os.path.splitext(video)[0]
        input_path = os.path.join(video_dir, video)

        # Output pattern: output_dir/videoName_%06d.png
        output_pattern = os.path.join(output_dir, f"{base_name}_%06d.png")

        # Construct and run ffmpeg command
        cmd = ["ffmpeg", "-i", input_path, "-vf", f"fps={fps}", output_pattern]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Frames extracted for {video} to {output_dir}")


if __name__ == "__main__":
    main()
