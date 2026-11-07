"""Export a Gaussian asset and render orbit and placement demos."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch


def unit(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Cannot infer the up direction. Specify --up X Y Z.")
    return vector / norm


def camera(eye, target, up, width, height):
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    forward = unit(target - eye)
    right = unit(np.cross(forward, up))
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    view = np.eye(4, dtype=np.float32)
    view[:3, :3], view[:3, 3] = rotation, -rotation @ eye
    view = torch.as_tensor(view.T.copy(), device="cuda")
    fovy = np.deg2rad(45)
    fovx = 2 * np.arctan(np.tan(fovy / 2) * width / height)
    projection = getProjectionMatrix(0.01, 100, fovx, fovy).T.cuda()
    return MiniCam(width, height, fovy, fovx, 0.01, 100, view, view @ projection)


def tensor(array):
    return torch.as_tensor(array, dtype=torch.float32, device="cuda")


def gaussian_stage():
    x, y = np.meshgrid(np.arange(-3, 3.01, 0.09), np.arange(-2.6, 2.61, 0.09))
    floor = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -0.012)))
    y, z = np.meshgrid(np.arange(-0.66, -0.39, 0.065), np.arange(0, 1.26, 0.065))
    wall = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    xyz = np.concatenate([floor, wall])
    colors = np.repeat([[0.48, 0.56, 0.62], [0.88, 0.46, 0.17]], [len(floor), len(wall)], axis=0)
    scales = np.repeat([[0.075, 0.075, 0.012], [0.11, 0.05, 0.05]], [len(floor), len(wall)], axis=0)
    rotations = np.tile([1, 0, 0, 0], (len(xyz), 1))
    return tensor(xyz), tensor(colors), tensor(scales), tensor(rotations)


def encode(path, count, draw):
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(count):
            rgb = draw(index / count).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            Image.fromarray(rgb).save(Path(tmp) / f"{index:05d}.png")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "24",
            "-i", str(Path(tmp) / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], check=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--up", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    sys.path.insert(0, str(args.upstream.resolve()))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.sh_utils import eval_sh

    ply = args.model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    model = GaussianModel(args.sh_degree)
    model.load_ply(str(ply))
    if args.up is None:
        cameras = json.loads((args.model / "cameras.json").read_text())
        up = unit(-np.mean(np.asarray([c["rotation"] for c in cameras])[:, :, 1], axis=0))
    else:
        up = unit(np.asarray(args.up))
    axis = np.eye(3)[np.argmin(np.abs(up))]
    right = unit(np.cross(axis, up))
    basis = np.stack([right, np.cross(up, right), up])
    xyz = model.get_xyz
    low, high = np.quantile(xyz.cpu().numpy() @ basis.T, [0.01, 0.99], axis=0)
    pivot = basis.T @ np.array([*(low[:2] + high[:2]) / 2, low[2]])
    height = float(high[2] - low[2])
    if height < 1e-6:
        raise ValueError("Degenerate asset height. Check reconstruction and up direction.")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ply, args.output / "asset.ply")
    (args.output / "asset.json").write_text(json.dumps({
        "ply": "asset.ply", "schema": "GraphDECO Gaussian PLY",
        "source_pivot": pivot.tolist(), "source_to_stage_rotation": basis.tolist(),
        "source_up": up.tolist(), "stage_up": [0, 0, 1], "scale": 1 / height,
        "units": "one reconstructed object height", "sh_degree": args.sh_degree,
    }, indent=2), encoding="utf-8")

    stage_xyz, stage_rgb, stage_scale, stage_rot = gaussian_stage()
    pivot_tensor = tensor(pivot)
    local_xyz = xyz - pivot_tensor
    sh = model.get_features.transpose(1, 2)
    source_q = model.get_rotation
    source_rotation = Rotation.from_quat(source_q[:, [1, 2, 3, 0]].cpu().numpy())
    pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False,
                           convert_SHs_python=False)
    background = tensor([0.12, 0.15, 0.19])

    def draw(eye, target, angle, translation, stage):
        matrix = Rotation.from_euler("z", angle).as_matrix() @ basis
        rotation = tensor(matrix)
        placed = local_xyz @ rotation.T / height + tensor(translation)
        direction = torch.nn.functional.normalize((placed - tensor(eye)) @ rotation, dim=1)
        rgb = (eval_sh(args.sh_degree, sh, direction) + 0.5).clamp_min(0)
        q = (Rotation.from_matrix(matrix) * source_rotation).as_quat()
        scales, rotations, opacity = model.get_scaling / height, tensor(q[:, [3, 0, 1, 2]]), model.get_opacity
        if stage:
            placed = torch.cat((placed, stage_xyz))
            scales = torch.cat((scales, stage_scale))
            rotations = torch.cat((rotations, stage_rot))
            opacity = torch.cat((opacity, torch.full_like(stage_xyz[:, :1], 0.97)))
            rgb = torch.cat((rgb, stage_rgb))
        scene = SimpleNamespace(get_xyz=placed, get_scaling=scales, get_rotation=rotations,
                                get_opacity=opacity, active_sh_degree=0)
        view = camera(np.asarray(eye), np.asarray(target), np.array([0, 0, 1]), 800, 600)
        return render(view, scene, pipe, background, override_color=rgb)["render"]

    encode(args.output / "orbit.mp4", args.frames, lambda t: draw(
        [2.6 * np.cos(2 * np.pi * t), 2.6 * np.sin(2 * np.pi * t), 1.3],
        [0, 0, 0.5], 0, [0, 0, 0], False))
    encode(args.output / "placement.mp4", args.frames, lambda t: draw(
        [0, -5.4, 2], [0, 0, 0.6], 2 * np.pi * t,
        [-1.5 + 3 * t, 0.2, 0], True))


if __name__ == "__main__":
    main()
