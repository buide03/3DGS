"""Sample a phone video and reconstruct its cameras with COLMAP."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def cpu_option(command, old, new):
    help_text = subprocess.run(
        ["colmap", command, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    ).stdout
    for option in (new, old):
        if option in help_text:
            return ["--" + option, "0"]
    raise RuntimeError(
        f"Cannot find --{new} or --{old} for COLMAP {command}.\n{help_text}"
    )


def extract_frames(video, output, count):
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 16:
            raise ValueError("The video must have at least 16 decodable frames")
        indices = np.linspace(0, total - 1, min(count, total), dtype=int)
        output.mkdir()
        for number, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or not cv2.imwrite(str(output / f"{number:05d}.jpg"), frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 98]):
                raise RuntimeError(f"Could not extract frame {index}")
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    if args.frames < 16:
        parser.error("--frames must be at least 16")
    if args.dataset.exists():
        parser.error("Choose a new dataset path to avoid mixing reconstructions")
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.dataset.parent) as work:
        work = Path(work)
        images, database, sparse = work / "input", work / "colmap.db", work / "models"
        extract_frames(args.video, images, args.frames)
        sparse.mkdir()
        run("colmap", "feature_extractor", "--database_path", database, "--image_path", images,
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
            *cpu_option("feature_extractor", "SiftExtraction.use_gpu", "FeatureExtraction.use_gpu"))
        run("colmap", "exhaustive_matcher", "--database_path", database,
            *cpu_option("exhaustive_matcher", "SiftMatching.use_gpu", "FeatureMatching.use_gpu"))
        run("colmap", "mapper", "--database_path", database, "--image_path", images, "--output_path", sparse)
        models = list(sparse.glob("*/images.bin"))
        if not models:
            raise RuntimeError("COLMAP did not reconstruct any cameras")
        def registered(path):
            with path.open("rb") as file:
                return int.from_bytes(file.read(8), "little")
        model = max(models, key=registered)
        print(f"COLMAP registered {registered(model)} / {len(list(images.iterdir()))} frames")
        if registered(model) < 16:
            raise RuntimeError("Too few registered cameras for training and holdout")
        dataset = work / "dataset"
        dataset.mkdir()
        run("colmap", "image_undistorter", "--image_path", images, "--input_path", model.parent,
            "--output_path", dataset, "--output_type", "COLMAP")
        model_dir = dataset / "sparse"
        binary_files = list(model_dir.glob("*.bin"))
        if binary_files:
            (model_dir / "0").mkdir()
            for file in binary_files:
                file.rename(model_dir / "0" / file.name)
        shutil.move(str(dataset), str(args.dataset))


if __name__ == "__main__":
    main()
