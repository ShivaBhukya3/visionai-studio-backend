#!/usr/bin/env python3
"""Download YOLOv8 models from Ultralytics hub."""
import sys
import argparse

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]


def download(model_names: list):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    for name in model_names:
        print(f"Downloading {name}...")
        try:
            YOLO(name)
            print(f"  ✓ {name} downloaded")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLOv8 models")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt"],
                        choices=MODELS, help="Models to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all models")
    args = parser.parse_args()

    targets = MODELS if args.all else args.models
    print(f"Downloading: {', '.join(targets)}")
    download(targets)
    print("Done!")
