import { createViewer } from '@playcanvas/supersplat-viewer/viewer';
import { GSPLATDATA_LARGE } from 'playcanvas';
import '@playcanvas/supersplat-viewer/viewer.css';
import { createGaussianOverlay } from './gaussian-overlay.js';
import './style.css';

const fileInput = document.querySelector('#file');
const stage = document.querySelector('#stage');
const container = document.querySelector('#viewer');
const welcome = document.querySelector('#welcome');
const status = document.querySelector('#status');
const frame = document.querySelector('#frame');
const open = document.querySelector('#open');
const choose = document.querySelector('#choose');
const renderMode = document.querySelector('#render-mode');
const pointsMode = document.querySelector('#points-mode');
const modelInfo = document.querySelector('#model-info');
const compare = document.querySelector('#compare');
const videoPanel = document.querySelector('#video-panel');
const videoFile = document.querySelector('#video-file');
const openVideo = document.querySelector('#open-video');
const video = document.querySelector('#source-video');
const videoStatus = document.querySelector('#video-status');
const videoInfo = document.querySelector('#video-info');

const settings = {
    version: 2,
    tonemapping: 'none',
    highPrecisionRendering: true,
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
let splat;
let overlay;
let contentUrl;
let videoUrl;
let loading = false;

function setFileLabel(element, label) {
    element.textContent = label;
    element.title = label;
}

function releaseScene() {
    overlay?.destroy();
    viewer?.destroy();
    viewer = undefined;
    splat = undefined;
    overlay = undefined;
    if (contentUrl) URL.revokeObjectURL(contentUrl);
    contentUrl = undefined;
    frame.disabled = true;
    renderMode.disabled = true;
    pointsMode.disabled = true;
    modelInfo.textContent = '';
    renderMode.setAttribute('aria-pressed', 'true');
    pointsMode.setAttribute('aria-pressed', 'false');
}

function setOverlayView(showOverlay) {
    if (!viewer?.state.loaded || !splat) return;
    if (showOverlay) overlay ??= createGaussianOverlay(viewer.app, splat);
    overlay?.setVisible(showOverlay);
    renderMode.setAttribute('aria-pressed', String(!showOverlay));
    pointsMode.setAttribute('aria-pressed', String(showOverlay));
    viewer.app.renderNextFrame = true;
}

async function loadScene(file) {
    if (!file || loading) return;
    if (!file.name.toLowerCase().endsWith('.ply') || file.size === 0) {
        setFileLabel(status, 'Hãy chọn một file Gaussian PLY đã xuất từ Nerfstudio.');
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
        setFileLabel(status, file.name);
        modelInfo.textContent = 'Đang đọc…';
        viewer = await createViewer({
            container,
            settings,
            contentUrl,
            contentFilename: 'scene.ply',
            renderer: 'webgl',
            ui: false,
            noanim: true,
        });
        const current = viewer;
        current.state.quality = 'high';
        const renderParams = current.app.scene.gsplat;
        // Keep fine splats and avoid the compact buffer's 8-bit scale and opacity.
        renderParams.dataFormat = GSPLATDATA_LARGE;
        renderParams.minPixelSize = 1;
        renderParams.colorUpdateAngle = 0.2;
        const asset = current.app.assets.getByUrl(contentUrl);
        const upAxis = header.match(/^comment\s+Vertical Axis:\s*([yz])\s*$/im)?.[1].toLowerCase() ?? 'z';
        const orientScene = () => {
            splat = current.app.root.findByName('gsplat');
            // This runs after the viewer creates its entity and before it frames the scene.
            // Nerfstudio's +Z becomes the viewer's +Y through a right-handed rotation.
            splat.setLocalEulerAngles(upAxis === 'z' ? -90 : 0, 0, 0);
        };
        if (asset.loaded) orientScene();
        else asset.once('load', orientScene);
        const ready = () => {
            if (viewer !== current || !current.state.loaded) return;
            current.frameScene();
            frame.disabled = false;
            renderMode.disabled = false;
            const count = splat?.gsplat.resource?.gsplatData?.numSplats;
            pointsMode.disabled = !count;
            modelInfo.textContent = `${(file.size / 1e6).toFixed(1)} MB${count ? ` · ${count.toLocaleString('vi-VN')} Gaussian` : ''}`;
        };
        current.events.on('progress:changed', progress => {
            if (viewer === current && !current.state.loaded) {
                modelInfo.textContent = `Đang đọc · ${Math.round(progress)}%`;
            }
        });
        current.events.once('loaded:changed', ready);
        if (current.state.loaded) ready();
    } catch (error) {
        if (!viewer) {
            releaseScene();
            welcome.hidden = false;
        }
        setFileLabel(status, `Không mở được mô hình. ${error.message}`);
        modelInfo.textContent = '';
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
renderMode.addEventListener('click', () => setOverlayView(false));
pointsMode.addEventListener('click', () => setOverlayView(true));
compare.addEventListener('click', () => {
    const comparing = videoPanel.hidden;
    videoPanel.hidden = !comparing;
    videoPanel.parentElement.classList.toggle('comparing', comparing);
    compare.setAttribute('aria-pressed', String(comparing));
    if (!comparing) video.pause();
});
openVideo.addEventListener('click', () => videoFile.click());
videoFile.addEventListener('change', () => {
    const file = videoFile.files[0];
    videoFile.value = '';
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(file);
    setFileLabel(videoStatus, file.name);
    videoInfo.textContent = 'Đang đọc…';
    video.hidden = false;
    video.src = videoUrl;
});
video.addEventListener('loadedmetadata', () => {
    const seconds = Math.floor(video.duration);
    const duration = Number.isFinite(seconds)
        ? ` · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
        : '';
    videoInfo.textContent = `${video.videoWidth} × ${video.videoHeight}${duration}`;
});
video.addEventListener('error', () => {
    if (!video.error) return;
    video.hidden = true;
    setFileLabel(videoStatus, 'Không phát được video');
    videoInfo.textContent = 'Thử MP4 (H.264) hoặc WebM';
});
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
    if (event.persisted) return;
    releaseScene();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
});
