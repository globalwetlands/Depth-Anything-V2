"""
This script reads a CSV file that lists video filenames, along with start and end times
for each clip to extract. It then uses ffmpeg to create these clips, organizing them
into directories based on the video filenames. By default, it copies the video segments
without re-encoding for faster, lossless extraction.
"""

import csv
import subprocess
import argparse
import os


def time_to_seconds(timestr):
    """
    Convert a time string in the format "mm:ss" to the total number of seconds.

    Parameters
    ----------
    timestr : str
        The time string, e.g. "01:30" for 1 minute and 30 seconds.

    Returns
    -------
    int
        The total seconds represented by the input time.
    """
    parts = timestr.split(":")
    if len(parts) != 2:
        raise ValueError(f"Time format should be mm:ss, got {timestr}")
    minutes, seconds = parts
    return int(minutes) * 60 + int(seconds)


def main():
    """
    Extract video clips defined in a CSV file using ffmpeg.

    The CSV should have columns:
    - id: a unique identifier for the clip
    - filename: the base name of the source video (without extension)
    - start-time: start time in "mm:ss"
    - end-time: end time in "mm:ss"

    This script:
    1. Reads the CSV and for each entry:
       - Converts start/end times to seconds.
       - Creates a folder named after the video filename.
       - Extracts the specified clip and saves it as "filename_id.mp4".
    2. By default, the video is copied without re-encoding, for faster and lossless extraction.

    Command-line arguments:
    --csv_file : Path to the CSV file.
    --video_dir : Directory containing the source videos (default: current directory).
    --output_dir : Directory to store extracted clips (default: current directory).
    """
    parser = argparse.ArgumentParser(
        description="Extract video clips based on CSV start/end times without re-encoding."
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=False,
        default=".",
        help="Directory containing the input videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=".",
        help="Directory to save the extracted clips.",
    )

    args = parser.parse_args()

    csv_file = args.csv_file
    video_dir = args.video_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["id"]
            filename = row["filename"]
            start_str = row["start-time"]
            end_str = row["end-time"]

            start_seconds = time_to_seconds(start_str)
            end_seconds = time_to_seconds(end_str)

            output_folder = os.path.join(output_dir, filename)
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, f"{filename}_{clip_id}.mp4")
            if os.path.exists(output_path):
                print(f"Skipping {output_path}: file already exists.")
                continue

            cmd = [
                "ffmpeg",
                "-i",
                os.path.join(video_dir, f"{filename}.MP4"),
                "-ss",
                str(start_seconds),
                "-to",
                str(end_seconds),
                "-c",
                "copy",
                output_path,
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
