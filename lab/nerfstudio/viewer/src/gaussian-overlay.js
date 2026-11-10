import {
    BLEND_NORMAL, Color, Entity, Layer, Mat4, Mesh, MeshInstance, PRIMITIVE_POINTS,
    Quat, SEMANTIC_ATTR10, SEMANTIC_ATTR11, SEMANTIC_ATTR12, SEMANTIC_ATTR14, SEMANTIC_ATTR15,
    SEMANTIC_COLOR, SEMANTIC_POSITION, ShaderMaterial, TYPE_FLOAT32, Vec3, VertexBuffer, VertexFormat,
} from 'playcanvas';

const projectionShader = `
    attribute vec3 vertex_position;
    attribute vec4 axis_x;
    attribute vec4 axis_y;
    attribute vec4 axis_z;
    attribute vec4 center;
    attribute vec4 splat_color;
    uniform mat4 matrix_model;
    uniform mat4 matrix_viewProjection;
    varying vec4 gaussianColor;
    varying vec2 ellipseUV;

    vec2 projectAxis(vec4 axis, vec4 clipCenter) {
        return (axis.xy - clipCenter.xy * (axis.w / clipCenter.w)) / clipCenter.w;
    }

    void main(void) {
        mat4 toClip = matrix_viewProjection * matrix_model;
        vec4 clipCenter = toClip * center;
        vec4 clipX = toClip * axis_x;
        vec4 clipY = toClip * axis_y;
        vec4 clipZ = toClip * axis_z;
        float depthRadius = 2.0 * length(vec3(clipX.w, clipY.w, clipZ.w));
        gaussianColor = splat_color;
        ellipseUV = vertex_position.xy;
        // A 2σ contour crossing the camera plane has no stable projected ellipse.
        if (clipCenter.w <= max(depthRadius, 0.0001)) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            return;
        }
        vec2 x = projectAxis(clipX, clipCenter);
        vec2 y = projectAxis(clipY, clipCenter);
        vec2 z = projectAxis(clipZ, clipCenter);
        // Factor the 2D covariance JΣJᵀ to map a unit circle to its 2σ ellipse.
        vec3 rowX = vec3(x.x, y.x, z.x);
        vec3 rowY = vec3(x.y, y.y, z.y);
        float a = sqrt(max(dot(rowX, rowX), 1e-12));
        float b = dot(rowX, rowY) / a;
        float c = sqrt(max(dot(rowY, rowY) - b * b, 0.0));
        vec2 offset = 2.0 * mat2(a, b, 0.0, c) * ellipseUV;
        gl_Position = clipCenter + vec4(offset * clipCenter.w, 0.0, 0.0);
    }
`;

export function createGaussianOverlay(app, splat) {
    const data = splat.gsplat.resource.gsplatData;
    const count = data.numSplats;
    const instances = new Float32Array(count * 20);
    const centers = new Float32Array(count * 3);
    const colors = new Uint8Array(count * 4);
    const position = new Vec3();
    const color = new Color();
    const rotation = new Quat();
    const scale = new Vec3();
    const transform = new Mat4();
    // The engine's PLY iterator decodes log-scale and the SH base color.
    const iterator = data.createIter(position, rotation, scale, color);
    const toByte = value => Math.round(Math.min(1, Math.max(0, value)) * 255);
    for (let i = 0; i < count; i++) {
        iterator.read(i);
        const p = i * 3;
        centers[p] = position.x;
        centers[p + 1] = position.y;
        centers[p + 2] = position.z;
        const c = i * 4;
        colors[c] = toByte(color.r);
        colors[c + 1] = toByte(color.g);
        colors[c + 2] = toByte(color.b);
        colors[c + 3] = 255;
        const offset = i * 20;
        instances.set(transform.setTRS(position, rotation.normalize(), scale).data, offset);
        instances[offset + 16] = color.r;
        instances[offset + 17] = color.g;
        instances[offset + 18] = color.b;
    }

    const device = app.graphicsDevice;
    const outlineMesh = new Mesh(device);
    outlineMesh.setPositions([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]);
    outlineMesh.setIndices([0, 1, 2, 0, 2, 3]);
    outlineMesh.update();
    const outlineMaterial = new ShaderMaterial({
        uniqueName: 'GaussianOutlines',
        vertexGLSL: projectionShader,
        attributes: {
            vertex_position: SEMANTIC_POSITION,
            axis_x: SEMANTIC_ATTR11,
            axis_y: SEMANTIC_ATTR12,
            axis_z: SEMANTIC_ATTR14,
            center: SEMANTIC_ATTR15,
            splat_color: SEMANTIC_ATTR10,
        },
        fragmentGLSL: `
            varying vec4 gaussianColor;
            varying vec2 ellipseUV;
            void main(void) {
                float radius = length(ellipseUV);
                float pixel = max(fwidth(radius), 0.0001);
                float distanceInPixels = (1.0 - radius) / pixel;
                float coverage = clamp(distanceInPixels + 0.5, 0.0, 1.0)
                               - clamp(distanceInPixels - 0.5, 0.0, 1.0);
                if (coverage <= 0.0) discard;
                gl_FragColor = vec4(clamp(gaussianColor.rgb, 0.0, 1.0), 0.45 * coverage);
            }
        `,
    });
    outlineMaterial.blendType = BLEND_NORMAL;
    outlineMaterial.depthTest = false;
    outlineMaterial.depthWrite = false;

    const centerMesh = new Mesh(device);
    centerMesh.setPositions(centers);
    centerMesh.setColors32(colors);
    centerMesh.update(PRIMITIVE_POINTS);
    const centerMaterial = new ShaderMaterial({
        uniqueName: 'GaussianCenters',
        attributes: { vertex_position: SEMANTIC_POSITION, vertex_color: SEMANTIC_COLOR },
        vertexGLSL: `
            attribute vec3 vertex_position;
            attribute vec4 vertex_color;
            uniform mat4 matrix_model;
            uniform mat4 matrix_viewProjection;
            varying vec3 gaussianColor;
            void main(void) {
                gl_Position = matrix_viewProjection * matrix_model * vec4(vertex_position, 1.0);
                gl_PointSize = 1.0;
                gaussianColor = vertex_color.rgb;
            }
        `,
        fragmentGLSL: `
            varying vec3 gaussianColor;
            void main(void) { gl_FragColor = vec4(gaussianColor * 0.72, 1.0); }
        `,
    });

    const entity = new Entity('Gaussian overlay', app);
    const centerEntity = new Entity('Gaussian centers', app);
    const outlines = new MeshInstance(outlineMesh, outlineMaterial, entity);
    const dots = new MeshInstance(centerMesh, centerMaterial, centerEntity);
    const format = new VertexFormat(device,
        [SEMANTIC_ATTR11, SEMANTIC_ATTR12, SEMANTIC_ATTR14, SEMANTIC_ATTR15, SEMANTIC_ATTR10]
            .map(semantic => ({ semantic, components: 4, type: TYPE_FLOAT32 })));
    const instanceBuffer = new VertexBuffer(device, format, count, { data: instances });
    outlines.setInstancing(instanceBuffer);
    outlines.pick = false;
    dots.pick = false;

    const outlineLayer = new Layer({ name: 'Gaussian outlines' });
    const centerLayer = new Layer({ name: 'Gaussian centers', clearDepthBuffer: true });
    app.scene.layers.push(outlineLayer);
    app.scene.layers.push(centerLayer);
    const camera = app.root.findByName('camera').camera;
    camera.layers = [...camera.layers, outlineLayer.id, centerLayer.id];
    entity.addComponent('render', { layers: [outlineLayer.id], meshInstances: [outlines] });
    centerEntity.addComponent('render', { layers: [centerLayer.id], meshInstances: [dots] });
    entity.addChild(centerEntity);
    splat.addChild(entity);

    return {
        setVisible(visible) {
            outlineLayer.enabled = visible;
            centerLayer.enabled = visible;
            entity.enabled = visible;
        },
        destroy() {
            entity.destroy();
            instanceBuffer.destroy();
            outlineMaterial.destroy();
            centerMaterial.destroy();
            camera.layers = camera.layers.filter(id => id !== outlineLayer.id && id !== centerLayer.id);
            app.scene.layers.remove(outlineLayer);
            app.scene.layers.remove(centerLayer);
        },
    };
}
