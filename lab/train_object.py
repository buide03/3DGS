"""Add foreground supervision to the pinned GraphDECO trainer."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from PIL import Image


@lru_cache(maxsize=256)
def read_mask(folder, name):
    with Image.open(folder / f"{Path(name).stem}.png") as image:
        mask = np.asarray(image.convert("L")) >= 128
    if not mask.any() or mask.all():
        raise ValueError(f"{name}: mask must contain both foreground and background")
    return mask


def filter_points(points, cameras, folder):
    visible = np.zeros(len(points), dtype=np.int32)
    foreground = np.zeros_like(visible)
    for camera in cameras:
        mask = read_mask(folder, camera.image_name)
        h, w = mask.shape
        if (w, h) != (camera.width, camera.height):
            raise ValueError(f"{camera.image_name}: mask must match the undistorted image size")
        xyz = points @ camera.R + camera.T
        ids = np.flatnonzero(xyz[:, 2] > 0.01)
        x, y, z = xyz[ids].T
        u = (x / z / np.tan(camera.FovX / 2) + 1) * w / 2
        v = (y / z / np.tan(camera.FovY / 2) + 1) * h / 2
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids = ids[valid]
        visible[ids] += 1
        foreground[ids] += mask[v[valid].astype(int), u[valid].astype(int)]
    keep = (foreground >= 2) & (foreground >= 0.7 * visible)
    if keep.sum() < 4:
        raise ValueError("Too few foreground SfM points. Check masks and COLMAP coverage.")
    print(f"Foreground initialization: {keep.sum()} / {len(points)} points")
    return keep


def foreground_loss(camera, model, pipe, image, target, ssim, folder):
    import torch
    from gaussian_renderer import render

    raw = read_mask(folder, camera.image_name)
    mask = torch.from_numpy(raw).to(device=image.device, dtype=image.dtype)[None, None]
    mask = torch.nn.functional.interpolate(mask, size=image.shape[-2:], mode="nearest")[0]
    # White splats over black produce accumulated alpha using the same rasterizer.
    alpha = render(camera, model, pipe, image.new_zeros(3),
                   override_color=torch.ones_like(model.get_xyz))["render"][:1]
    rgb_loss = ((image - target).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)
    silhouette_loss = (alpha - mask).abs().mean()
    return rgb_loss, ssim(image * mask, target * mask), silhouette_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    folder = args.masks.resolve()
    if not folder.is_dir():
        parser.error("--masks must point to a folder of foreground PNG files")
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    from scene.dataset_readers import sceneLoadTypeCallbacks

    original_loader = sceneLoadTypeCallbacks["Colmap"]

    def load_foreground(*loader_args, **kwargs):
        info = original_loader(*loader_args, **kwargs)
        cloud = info.point_cloud
        if cloud is None:
            raise ValueError("COLMAP did not supply a sparse point cloud")
        keep = filter_points(cloud.points, info.train_cameras, folder)
        return info._replace(point_cloud=cloud._replace(
            points=cloud.points[keep], colors=cloud.colors[keep], normals=cloud.normals[keep]
        ))

    sceneLoadTypeCallbacks["Colmap"] = load_foreground
    train = upstream / "train.py"
    source = train.read_text(encoding="utf-8")
    start = "        if viewpoint_cam.alpha_mask is not None:"
    end = "        # Depth regularization"
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError("Unsupported trainer. Use the pinned GraphDECO revision.")
    first, last = source.index(start), source.index(end)
    source = source[:first] + """        gt_image = viewpoint_cam.original_image.cuda()
        Ll1, ssim_value, silhouette_loss = lab_loss(
            viewpoint_cam, gaussians, pipe, image, gt_image, ssim
        )
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss += 0.2 * silhouette_loss

""" + source[last:]
    sys.argv = [str(train), *upstream_args]
    exec(compile(source, str(train), "exec"), {
        "__name__": "__main__", "__file__": str(train),
        "lab_loss": lambda *values: foreground_loss(*values, folder),
    })


if __name__ == "__main__":
    main()
