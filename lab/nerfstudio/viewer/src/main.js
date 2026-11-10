import { createViewer } from '@playcanvas/supersplat-viewer/viewer';
import '@playcanvas/supersplat-viewer/viewer.css';
import './style.css';

const fileInput = document.querySelector('#file');
const stage = document.querySelector('#stage');
const container = document.querySelector('#viewer');
const welcome = document.querySelector('#welcome');
const status = document.querySelector('#status');
const frame = document.querySelector('#frame');
const open = document.querySelector('#open');
const choose = document.querySelector('#choose');

const settings = {
    version: 2,
    tonemapping: 'none',
    highPrecisionRendering: false,
    background: { color: [0.055, 0.075, 0.09] },
    postEffectSettings: {
        sharpness: { enabled: false, amount: 0 },
        bloom: { enabled: false, intensity: 0.1, blurLevel: 2 },
        grading: { enabled: false, brightness: 1, contrast: 1, saturation: 1, tint: [1, 1, 1] },
        vignette: { enabled: false, intensity: 0.5, inner: 0.3, outer: 0.75, curvature: 1 },
        fringing: { enabled: false, intensity: 0.5 },
    },
    animTracks: [],
    cameras: [],
    annotations: [],
    startMode: 'default',
};

let viewer;
let contentUrl;
let loading = false;

function releaseScene() {
    viewer?.destroy();
    viewer = undefined;
    if (contentUrl) URL.revokeObjectURL(contentUrl);
    contentUrl = undefined;
    frame.disabled = true;
}

async function loadScene(file) {
    if (!file || loading) return;
    if (!file.name.toLowerCase().endsWith('.ply') || file.size === 0) {
        status.textContent = 'Hãy chọn một file Gaussian PLY đã xuất từ Nerfstudio.';
        return;
    }
    loading = true;
    open.disabled = true;
    choose.disabled = true;
    try {
        const header = await file.slice(0, 8192).text();
        if (!/^ply\r?\n/.test(header) || !header.includes('end_header')) {
            throw new Error('File không có phần đầu PLY hợp lệ.');
        }
        releaseScene();
        welcome.hidden = true;
        contentUrl = URL.createObjectURL(file);
        status.textContent = `Đang đọc ${file.name}…`;
        viewer = await createViewer({
            container,
            settings,
            contentUrl,
            contentFilename: 'scene.ply',
            renderer: 'webgl',
        });
        const current = viewer;
        const ready = () => {
            if (viewer !== current || !current.state.loaded) return;
            current.frameScene();
            frame.disabled = false;
            status.textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
        };
        current.events.on('progress:changed', progress => {
            if (viewer === current && !current.state.loaded) {
                status.textContent = `Đang đọc ${file.name} · ${Math.round(progress)}%`;
            }
        });
        current.events.once('loaded:changed', ready);
        if (current.state.loaded) ready();
    } catch (error) {
        if (!viewer) {
            releaseScene();
            welcome.hidden = false;
        }
        status.textContent = `Không mở được mô hình. ${error.message}`;
    } finally {
        loading = false;
        open.disabled = false;
        choose.disabled = false;
    }
}

open.addEventListener('click', () => fileInput.click());
choose.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    loadScene(fileInput.files[0]);
    fileInput.value = '';
});
frame.addEventListener('click', () => viewer?.frameScene());
stage.addEventListener('dragover', event => {
    event.preventDefault();
    stage.classList.add('dragging');
});
stage.addEventListener('dragleave', event => {
    if (!stage.contains(event.relatedTarget)) stage.classList.remove('dragging');
});
stage.addEventListener('drop', event => {
    event.preventDefault();
    stage.classList.remove('dragging');
    loadScene(event.dataTransfer.files[0]);
});
window.addEventListener('pagehide', event => {
    if (!event.persisted) releaseScene();
});
