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
    from ipyevents import Event

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
    azimuth = widgets.FloatSlider(description="Góc ngang", min=-180, max=180, step=1)
    elevation = widgets.FloatSlider(description="Góc cao", min=-75, max=75, value=15, step=1)
    zoom = widgets.FloatSlider(description="Zoom", min=0.5, max=3, value=1, step=0.05)
    button = widgets.Button(description="Render góc nhìn", button_style="primary")
    reset = widgets.Button(description="Đặt lại góc nhìn")
    width, height = 640, 480
    preview = widgets.Image(format="png", width=width, height=height)
    instructions = widgets.HTML("Kéo chuột trái để xoay · Cuộn để zoom · Shift + kéo để dịch khung nhìn")
    status = widgets.Label()
    pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False,
                           convert_SHs_python=False)
    background = tensor([0.12, 0.15, 0.19])
    state = {}

    drag = None

    def camera_direction():
        up, right = state["up"], state["right"]
        yaw, pitch = np.deg2rad([azimuth.value, elevation.value])
        horizontal = np.cos(yaw) * right + np.sin(yaw) * np.cross(up, right)
        return np.cos(pitch) * horizontal + np.sin(pitch) * up

    def camera_distance():
        return 1.6 * state["radius"] / zoom.value

    @torch.no_grad()
    def redraw(_=None):
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
                             up=up, right=unit(right), pan=np.zeros(3))

            center = state["center"] + state["pan"]
            eye = center + camera_distance() * camera_direction()
            view = camera(eye, center, state["up"], width, height)
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

    def reset_view(_):
        nonlocal drag
        drag = None
        if state:
            state["pan"].fill(0)
        azimuth.value, elevation.value, zoom.value = 0, 15, 1
        redraw()

    def drag_start_or_end(event):
        nonlocal drag
        if event["type"] == "mousedown" and event["button"] == 0:
            drag = (event["clientX"], event["clientY"])
        else:
            drag = None

    def move_or_zoom(event):
        nonlocal drag
        if not state or state.get("ply") != selected.value:
            return
        if event["type"] == "wheel":
            delta = event["deltaY"]
            if delta:
                factor = 1.12 if delta < 0 else 1 / 1.12
                zoom.value = float(np.clip(zoom.value * factor, zoom.min, zoom.max))
                redraw()
            return
        if drag is None or not (event.get("buttons", 0) & 1):
            return
        x, y = event["clientX"], event["clientY"]
        dx, dy = x - drag[0], y - drag[1]
        drag = (x, y)
        if not (dx or dy):
            return
        if event.get("shiftKey", False):
            forward = -camera_direction()
            right = unit(np.cross(forward, state["up"]))
            down = np.cross(forward, right)
            units_per_pixel = 2 * camera_distance() * np.tan(np.deg2rad(22.5)) / height
            state["pan"] -= (dx * right + dy * down) * units_per_pixel
        else:
            azimuth.value = (azimuth.value - 0.4 * dx + 180) % 360 - 180
            elevation.value = float(np.clip(
                elevation.value + 0.4 * dy, elevation.min, elevation.max
            ))
        redraw()

    drag_events = Event(source=preview, watched_events=["mousedown", "mouseup", "mouseleave", "dragstart"],
                        prevent_default_action=True)
    motion_events = Event(source=preview, watched_events=["mousemove", "wheel"],
                          prevent_default_action=True, wait=150)
    drag_events.on_dom_event(drag_start_or_end)
    motion_events.on_dom_event(move_or_zoom)
    selected.observe(reset_view, names="value")
    reset.on_click(reset_view)
    button.on_click(redraw)
    display(widgets.VBox([
        selected, azimuth, elevation, zoom, widgets.HBox([button, reset]),
        instructions, status, preview,
    ]))
    redraw()
