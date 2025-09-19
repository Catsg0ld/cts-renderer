/**
 * @fileoverview Application file for CTSRenderer triangle demo.
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
 * Main application function.
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

    let aspect;

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

    const triangleVertices = new Float32Array([
      0.0, 0.5, -0.5, -0.5, 0.5, -0.5,
    ]);

    renderer.createVertexBuffer('triangle-vertices', triangleVertices);

    // Shader setup
    const shaders = {
      vertex: `
          
          struct Uniforms {
            projection: mat4x4<f32>,
          }

          @group(0) @binding(0) var<uniform> uniforms: Uniforms;

          @vertex
          fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
            return uniforms.projection * vec4<f32>(position, 0.0, 1.0);
          }
        `,
      fragment: `
          @fragment
          fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); 
          }
        `,
    };

    // Define vertex buffer layout
    const vertexBufferLayout = {
      arrayStride: 8, // 2 float32 * 4 bytes each = 8 bytes
      attributes: [
        {
          shaderLocation: 0,
          offset: 0,
          format: 'float32x2',
        },
      ],
    };

    // Bind group layout for uniform buffer
    const bindGroupLayout = renderer.createBindGroupLayout('triangle-layout', [
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

    const view = mat4.lookAt(eye, target, up);
    const projection = mat4.perspective(rad(45), aspect, 0.1, 1000);
    const matrixData = mat4.multiply(projection, view);

    const uniformBuffer = renderer.createUniformBuffer(
      'triangle-uniforms',
      matrixData
    );

    renderer.setMSAA(true);

    renderer.createPipeline('triangle-pipeline', {
      shaders: shaders,
      vertexBuffers: [vertexBufferLayout],
      bindGroupLayouts: [bindGroupLayout],
      primitive: {
        topology: 'triangle-list',
      },
    });

    renderer.setPipeline('triangle-pipeline');

    renderer.createBindGroup({
      name: 'triangle-bind-group',
      entries: [
        {
          binding: 0,
          resource: { buffer: uniformBuffer },
        },
      ],
    });

    // Finally render
    var lastTime = 0;
    var angle = 0;
    function render(currentTime) {
      const deltaTime = (currentTime - lastTime) / 1000;
      lastTime = currentTime;
      angle = (angle + 15 * deltaTime) % 360;

      const view = mat4.lookAt(eye, target, up);
      const projection = mat4.perspective(rad(45), aspect, 0.1, 1000);
      const matrixData = mat4.rotateY(mat4.multiply(projection, view), angle);
      renderer.updateBuffer('triangle-uniforms', matrixData);
      renderer.render({
        vertexBuffers: {
          0: 'triangle-vertices', // Set vertex buffer to slot 0
        },
        bindGroups: { 0: 'triangle-bind-group' },
        vertexCount: 3, // 3 for a triangle
      });
      requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
  } catch (error) {
    console.error('Failed to initialize application:', error);

    // Show user-friendly error message (you can modify it btw)
    document.body.innerHTML = HTMLWGPUErrorMessage(error);
  }
}

/**
 * Utility function to check WebGPU support
 * @returns {boolean}
 */
function checkWebGPUSupport() {
  const supported = WGPURenderer.isSupported();

  if (!supported) {
    document.body.innerHTML = HTMLWGPUUnsupportedMessage(
      'Bro buy a new computer😭🙏'
    );
    return false;
  }
  return true;
}

// Start application when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (checkWebGPUSupport()) {
      main();
    }
  });
} else {
  if (checkWebGPUSupport()) {
    main();
  }
}
