"""Inspect saved GraphDECO models using Colab widgets and its CUDA renderer."""

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from demo import camera, tensor, unit


def show_viewer(upstream, model_dirs, sh_degree=3):
    import ipywidgets as widgets
    from IPython.display import display

    sys.path.insert(0, str(Path(upstream).resolve()))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    choices = []
    for label, folder in model_dirs.items():
        for ply in Path(folder).glob("point_cloud/iteration_*/point_cloud.ply"):
            iteration = ply.parent.name.removeprefix("iteration_")
            if iteration.isdigit():
                choices.append((label, int(iteration), ply))
    if not choices:
        raise FileNotFoundError("Chưa có mô hình PLY. Hãy chạy cell train trước.")
    choices.sort(key=lambda entry: (entry[0], -entry[1]))

    selected = widgets.Dropdown(
        options=[(f"{label} · vòng {iteration}", ply) for label, iteration, ply in choices],
        description="Mô hình",
    )
    azimuth = widgets.IntSlider(description="Góc ngang", min=-180, max=180, step=5)
    elevation = widgets.IntSlider(description="Góc cao", min=-75, max=75, value=15, step=5)
    zoom = widgets.FloatSlider(description="Zoom", min=0.5, max=3, value=1, step=0.1)
    button = widgets.Button(description="Render góc nhìn", button_style="primary")
    preview = widgets.Image(format="png", width=640, height=480)
    status = widgets.Label()
    pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False,
                           convert_SHs_python=False)
    background = tensor([0.12, 0.15, 0.19])
    state = {}

    @torch.no_grad()
    def redraw(_):
        button.disabled = True
        status.value = "Đang render trên GPU Colab…"
        try:
            ply = selected.value
            if state.get("ply") != ply:
                state.clear()
                model = GaussianModel(sh_degree)
                model.load_ply(str(ply))
                points = model.get_xyz.cpu().numpy()
                lo, hi = np.quantile(points, [0.05, 0.95], axis=0)
                center = (lo + hi) / 2
                radius = max(float(np.linalg.norm(hi - lo)), 0.02)
                up = np.array([0., 0., 1.])
                right = np.array([1., 0., 0.])
                cameras_file = ply.parents[2] / "cameras.json"
                if cameras_file.is_file():
                    cameras = json.loads(cameras_file.read_text(encoding="utf-8"))
                    if cameras:
                        avg_up = -np.mean(np.asarray([c["rotation"] for c in cameras])[:, :, 1], axis=0)
                        if np.linalg.norm(avg_up) > 1e-6:
                            up = unit(avg_up)
                        direction = np.asarray(cameras[0]["position"]) - center
                        right = direction - np.dot(direction, up) * up
                right = right - np.dot(right, up) * up
                if np.linalg.norm(right) < 1e-6:
                    right = np.cross(np.eye(3)[np.argmin(np.abs(up))], up)
                state.update(ply=ply, model=model, center=center, radius=radius,
                             up=up, right=unit(right))

            up, right = state["up"], state["right"]
            yaw, pitch = np.deg2rad([azimuth.value, elevation.value])
            horizontal = np.cos(yaw) * right + np.sin(yaw) * np.cross(up, right)
            direction = np.cos(pitch) * horizontal + np.sin(pitch) * up
            center = state["center"]
            eye = center + 1.6 * state["radius"] / zoom.value * direction
            view = camera(eye, center, up, 640, 480)
            rgb = render(view, state["model"], pipe, background)["render"]
            image = Image.fromarray(rgb.permute(1, 2, 0).mul(255).byte().cpu().numpy())
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                preview.value = buffer.getvalue()
            status.value = f"Đã render {state['model'].get_xyz.shape[0]:,} Gaussian."
        except Exception as exc:
            preview.value = b""
            status.value = f"Lỗi render: {exc}"
        finally:
            button.disabled = False

    button.on_click(redraw)
    display(widgets.VBox([selected, azimuth, elevation, zoom, button, status, preview]))
    redraw(None)
