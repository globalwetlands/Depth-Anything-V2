import csv
import subprocess
import argparse
import os


def time_to_seconds(timestr):
    """Convert mm:ss to total seconds."""
    parts = timestr.split(":")
    if len(parts) != 2:
        raise ValueError(f"Time format should be mm:ss, got {timestr}")
    minutes, seconds = parts
    return int(minutes) * 60 + int(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Clip videos based on CSV start/end times, creating folders for each filename."
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to the CSV file."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=False,
        default=".",
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=".",
        help="Directory to save clipped videos.",
    )
    parser.add_argument(
        "--no_reencode", action="store_true", help="Use -c copy to avoid re-encoding."
    )

    args = parser.parse_args()

    csv_file = args.csv_file
    video_dir = args.video_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV and process each line
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row["id"]
            filename = row["filename"]
            start_str = row["start-time"]
            end_str = row["end-time"]

            # Convert times to seconds
            start_seconds = time_to_seconds(start_str)
            end_seconds = time_to_seconds(end_str)

            # Create folder based on filename
            output_folder = os.path.join(output_dir, filename)
            os.makedirs(output_folder, exist_ok=True)

            # Build final output path
            output_path = os.path.join(output_folder, f"{filename}_{clip_id}.mp4")

            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {output_path}: file already exists.")
                continue

            # Construct the ffmpeg command
            cmd = [
                "ffmpeg",
                "-i",
                os.path.join(video_dir, f"{filename}.MP4"),
                "-ss",
                str(start_seconds),
                "-to",
                str(end_seconds),
            ]

            # If no re-encode is desired (the input format must be compatible)
            if args.no_reencode:
                cmd += ["-c", "copy"]
            else:
                # Otherwise, re-encode (you can choose codecs if needed)
                cmd += ["-c:v", "libx264", "-c:a", "aac", "-strict", "experimental"]

            cmd += [output_path]

            # Run the ffmpeg command
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
