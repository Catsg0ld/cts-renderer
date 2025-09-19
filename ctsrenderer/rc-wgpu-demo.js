/**
 * @fileoverview CTSRenderer cube demo with WGPURenderer.
 * This demo renders a rotating cube using CTSRenderer.
 */

import {
  WGPURenderer,
  HTMLWGPUErrorMessage,
  HTMLWGPUUnsupportedMessage,
} from './CTSRenderer.js';

import { mat4 } from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.min.js';

/* Uncomment to simulate WebGPU error for debugging
Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
*/

/**
 * Converts degrees to radians.
 * @param {number} deg - Angle in degrees.
 * @returns {number} Angle in radians.
 */
function rad(deg) {
  return (deg * Math.PI) / 180;
}

/**
 * Main application entry point.
 * Initializes WebGPU, sets up buffers, shaders, pipeline, and starts render loop.
 */
async function main() {
  const canvas = document.getElementById('gpu-canvas');
  if (!canvas) {
    console.error('Canvas element not found!');
    return;
  }

  try {
    // Renderer initialization
    const renderer = new WGPURenderer(canvas);
    await renderer.initialize();

    let aspect = 0;

    /**
     * Resizes canvas and updates renderer viewport.
     */
    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      renderer.setSize(canvas.width, canvas.height);
      aspect = canvas.width / canvas.height;
    }
    window.addEventListener('resize', resize);
    resize();

    const cubeVertices = new Float32Array([
      -0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.5, -0.5, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5,
      0.5, 0.0, 1.0, 1.0, -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,

      -0.5, -0.5, -0.5, 0.0, 1.0, 0.0, -0.5, 0.5, -0.5, 1.0, 0.0, 1.0, 0.5, 0.5,
      -0.5, 0.0, 1.0, 0.0, 0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
    ]);

    const cubeIndices = new Uint16Array([
      0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 3, 2, 6, 6, 5, 3, 0, 4, 7, 7, 1, 0, 1,
      7, 6, 6, 2, 1, 0, 3, 5, 5, 4, 0,
    ]);

    renderer.createVertexBuffer('cube-vertices', cubeVertices);
    renderer.createIndexBuffer('cube-indices', cubeIndices);

    // Shader setup
    const shaders = {
      vertex: `
        struct Uniforms {
          mvpMatrix: mat4x4<f32>,
        }

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) color: vec4<f32>
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @vertex
        fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
          var output: VertexOutput;
          output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
          output.color = vec4<f32>(color, 1.0);
          return output;
        }
      `,
      fragment: `
        @fragment
        fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
          return color;
        }
      `,
    };

    // Vertex buffer layout
    const vertexBufferLayout = {
      arrayStride: 24, // 6 float32 (24 bytes)
      attributes: [
        { format: 'float32x3', offset: 0, shaderLocation: 0 }, // position
        { format: 'float32x3', offset: 12, shaderLocation: 1 }, // color
      ],
    };

    // Bind group layout for uniform buffer
    const bindGroupLayout = renderer.createBindGroupLayout('cube-layout', [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      },
    ]);

    // Camera and projection setup
    const eye = [0, 0, 5];
    const target = [0, 0, 0];
    const up = [0, 1, 0];

    let angle = 0;
    const view = mat4.lookAt(eye, target, up);
    const projection = mat4.perspective(rad(45), aspect, 0.1, 1000);
    const matrixData = mat4.rotateY(mat4.multiply(projection, view), angle);

    // Create uniform buffer
    const uniformBuffer = renderer.createUniformBuffer(
      'cube-uniforms',
      matrixData
    );

    renderer.setMSAA(true);
    renderer.setDepthState(true);

    // Pipeline setup
    renderer.createPipeline('cube-pipeline', {
      shaders: shaders,
      vertexBuffers: [vertexBufferLayout],
      bindGroupLayouts: [bindGroupLayout],
      primitive: { topology: 'triangle-list' },
    });

    renderer.setPipeline('cube-pipeline');

    // Create bind group for uniforms
    renderer.createBindGroup({
      name: 'cube-bind-group',
      entries: [{ binding: 0, resource: uniformBuffer }],
    });

    // Render loop
    let lastTime = 0;
    function render(time) {
      const deltaTime = (time - lastTime) / 1000;
      lastTime = time;

      angle = (angle + 1 * deltaTime) % 360;

      const matrix = mat4.rotateX(
        mat4.rotateY(mat4.multiply(projection, view), angle),
        angle
      );

      renderer.updateBuffer('cube-uniforms', matrix);

      renderer.render({
        vertexBuffers: { 0: 'cube-vertices' },
        indexBuffer: 'cube-indices',
        bindGroups: { 0: 'cube-bind-group' },
        indexCount: cubeIndices.length,
        vertexCount: cubeVertices.length / 6,
      });

      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  } catch (error) {
    document.body.innerHTML = HTMLWGPUErrorMessage(error);
  }
}

/**
 * Checks if WGPURenderer is supported.
 * @returns {boolean} True if supported, false otherwise.
 */
function checkWGPURendererSupport() {
  if (!WGPURenderer.isSupported()) {
    document.body.innerHTML = HTMLWGPUUnsupportedMessage(
      'Bro buy a new computer🙏😭'
    );
    return false;
  }
  return true;
}

// Initialize App on DOM Ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (checkWGPURendererSupport()) main();
  });
} else {
  if (checkWGPURendererSupport()) main();
}
