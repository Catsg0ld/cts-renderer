/**
 * @fileoverview WebGPU API based renderer implementation.
 * @copyright 2025–present Catsgold
 * @license GPL-3.0-or-later
 * You must credit Catsgold (me) if you use or modify this code.
 */

import { Renderer } from './Renderer.js';

/**
 * Shader configuration object.
 * @typedef {Object} ShaderConfig
 * @property {string} vertex - Vertex shader code.
 * @property {string} fragment - Fragment shader code.
 */

/**
 * Buffer configuration object.
 * @typedef {Object} BufferConfig
 * @property {string} name - Buffer name.
 * @property {ArrayBuffer|ArrayBufferView} data - Buffer data.
 * @property {GPUBufferUsage} usage - Buffer usage flags.
 * @property {GPUIndexFormat} [indexFormat] - Index format for index buffers.
 */

/**
 * Pipeline configuration object.
 * @typedef {Object} PipelineConfig
 * @property {ShaderConfig} shaders - Shader configuration.
 * @property {Array<GPUVertexBufferLayout>} vertexBuffers - Vertex buffer layouts.
 * @property {GPUPrimitiveState} [primitive] - Primitive state configuration.
 * @property {GPUDepthStencilState} [depthStencil] - Depth stencil state.
 * @property {GPUBindGroupLayout[]} [bindGroupLayouts] - Explicit bind group layouts.
 */

/**
 * Bind group entry configuration.
 * @typedef {Object} BindGroupEntry
 * @property {number} binding - Binding point.
 * @property {GPUBuffer|GPUTextureView|GPUSampler} resource - Resource to bind.
 * @property {number} [size] - Size for buffer binding (if applicable).
 */

/**
 * Bind group configuration.
 * @typedef {Object} BindGroupConfig
 * @property {string} name - Bind group name.
 * @property {Array<BindGroupEntry>} entries - Bind group entries.
 */

/**
 * Texture configuration.
 * @typedef {Object} TextureConfig
 * @property {string} name - Texture name.
 * @property {number} width - Texture width.
 * @property {number} height - Texture height.
 * @property {GPUTextureFormat} [format='rgba8unorm'] - Texture format.
 * @property {GPUTextureUsage} [usage=GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST] - Texture usage.
 */

/**
 * Sampler configuration.
 * @typedef {Object} SamplerConfig
 * @property {string} name - Sampler name.
 * @property {GPUAddressMode} [addressModeU='clamp-to-edge'] - U address mode.
 * @property {GPUAddressMode} [addressModeV='clamp-to-edge'] - V address mode.
 * @property {GPUAddressMode} [addressModeW='clamp-to-edge'] - W address mode.
 * @property {GPUFilterMode} [magFilter='linear'] - Magnification filter.
 * @property {GPUFilterMode} [minFilter='linear'] - Minification filter.
 */

/**
 * Render target configuration.
 * @typedef {Object} RenderTargetConfig
 * @property {string} name - Render target name.
 * @property {number} width - Render target width.
 * @property {number} height - Render target height.
 * @property {GPUTextureFormat} [format='rgba8unorm'] - Texture format.
 * @property {number} [sampleCount=1] - Sample count for MSAA.
 */

/**
 * Event listener callback.
 * @callback EventCallback
 * @param {*} data - Event data.
 */

/**
 * Buffer metadata for better resource management.
 * @typedef {Object} BufferMetadata
 * @property {GPUBuffer} buffer - The GPU buffer.
 * @property {number} size - Buffer size in bytes.
 * @property {GPUBufferUsage} usage - Buffer usage flags.
 * @property {GPUIndexFormat} [indexFormat] - Index format for index buffers.
 */

/**
 * Texture metadata for better resource management.
 * @typedef {Object} TextureMetadata
 * @property {GPUTexture} texture - The GPU texture.
 * @property {GPUTextureView} view - The texture view.
 * @property {number} width - Texture width.
 * @property {number} height - Texture height.
 * @property {GPUTextureFormat} format - Texture format.
 * @property {GPUTextureUsage} usage - Texture usage.
 */

/**
 * Renderer using WebGPU API.
 * @extends Renderer
 */
export class WGPURenderer extends Renderer {
  /**
   * Creates an instance of WGPURenderer.
   * @param {HTMLCanvasElement} canvas - Canvas element for rendering.
   */
  constructor(canvas) {
    super(canvas);

    /**
     * GPU device.
     * @type {?GPUDevice}
     * @private
     */
    this.device_ = null;

    /**
     * WebGPU context.
     * @type {?GPUCanvasContext}
     * @private
     */
    this.context_ = null;

    /**
     * Texture format for rendering.
     * @type {?GPUTextureFormat}
     * @private
     */
    this.format_ = null;

    /**
     * Render pipelines.
     * @type {Map<string, GPURenderPipeline>}
     * @private
     */
    this.pipelines_ = new Map();

    /**
     * Compute pipelines.
     * @type {Map<string, GPUComputePipeline>}
     * @private
     */
    this.computePipelines_ = new Map();

    /**
     * GPU buffers with metadata.
     * @type {Map<string, BufferMetadata>}
     * @private
     */
    this.buffers_ = new Map();

    /**
     * Textures with metadata.
     * @type {Map<string, TextureMetadata>}
     * @private
     */
    this.textures_ = new Map();

    /**
     * Samplers.
     * @type {Map<string, GPUSampler>}
     * @private
     */
    this.samplers_ = new Map();

    /**
     * Bind groups.
     * @type {Map<string, GPUBindGroup>}
     * @private
     */
    this.bindGroups_ = new Map();

    /**
     * Render targets with metadata.
     * @type {Map<string, TextureMetadata>}
     * @private
     */
    this.renderTargets_ = new Map();

    /**
     * Current pipeline name.
     * @type {?string}
     * @private
     */
    this.currentPipeline_ = null;

    /**
     * Current clear color.
     * @type {GPUColor}
     * @private
     */
    this.clearColor_ = { r: 0.1, g: 0.1, b: 0.1, a: 1.0 };

    /**
     * Current clear depth.
     * @type {number}
     * @private
     */
    this.clearDepth_ = 1.0;

    /**
     * Current clear stencil.
     * @type {number}
     * @private
     */
    this.clearStencil_ = 0;

    /**
     * Event listeners.
     * @type {Map<string, Set<EventCallback>>}
     * @private
     */
    this.eventListeners_ = new Map();

    /**
     * Shader modules cache.
     * @type {Map<string, GPUShaderModule>}
     * @private
     */
    this.shaderModules_ = new Map();

    /**
     * Frame count.
     * @type {number}
     * @private
     */
    this.frameCount_ = 0;

    /**
     * Start time for statistics.
     * @type {number}
     * @private
     */
    this.startTime_ = performance.now();

    /**
     * Current render target.
     * @type {?string}
     * @private
     */
    this.currentRenderTarget_ = null;

    /**
     * Bind group layouts.
     * @type {Map<string, GPUBindGroupLayout>}
     * @private
     */
    this.bindGroupLayouts_ = new Map();

    /**
     * Current vertex buffers.
     * @type {Object<number, {buffer: GPUBuffer, offset: number}>}
     * @private
     */
    this.currentVertexBuffers_ = {};

    /**
     * Current bind groups.
     * @type {Object<number, GPUBindGroup>}
     * @private
     */
    this.currentBindGroups_ = {};

    /**
     * Blend state configuration.
     * @type {?GPUBlendState}
     * @private
     */
    this.blendState_ = null;

    /**
     * Depth stencil state configuration.
     * @type {?GPUDepthStencilState}
     * @private
     */
    this.depthStencilState_ = null;

    /**
     * Depth stencil texture metadata.
     * @type {?TextureMetadata}
     * @private
     */
    this.depthStencilTexture_ = null;

    /**
     * MSAA sample count.
     * @type {number}
     * @private
     */
    this.msaaSampleCount_ = 1;

    /**
     * MSAA texture metadata.
     * @type {?TextureMetadata}
     * @private
     */
    this.msaaTexture_ = null;

    /**
     * Blend state enabled.
     * @type {boolean}
     * @private
     */
    this.blendEnabled_ = false;

    /**
     * Depth state enabled.
     * @type {boolean}
     * @private
     */
    this.depthEnabled_ = false;

    /**
     * Current cull mode.
     * @type {GPUCullMode}
     * @private
     */
    this.cullMode_ = 'none';
  }

  /**
   * Initializes WebGPU renderer.
   * @override
   * @returns {Promise<boolean>} Promise that resolves to true on successful initialization.
   * @throws {Error} If WebGPU is not supported or initialization fails.
   */
  async initialize() {
    if (!WGPURenderer.isSupported()) {
      throw new Error('WebGPU is not supported in this browser.');
    }

    try {
      // Get adapter and device
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('Failed to get GPU adapter.');
      }

      this.device_ = await adapter.requestDevice();
      if (!this.device_) {
        throw new Error('Failed to get GPU device.');
      }

      // Configure canvas context
      this.context_ = this.canvas.getContext('webgpu');
      if (!this.context_) {
        throw new Error('Failed to get WebGPU context.');
      }

      this.format_ = navigator.gpu.getPreferredCanvasFormat();
      this.context_.configure({
        device: this.device_,
        format: this.format_,
        alphaMode: 'premultiplied',
      });

      this.isInitialized = true;
      this.emit('initialized', { device: this.device_, format: this.format_ });
      return true;
    } catch (error) {
      console.error('Failed to initialize WGPURenderer:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Destroys a buffer and removes it from management.
   * @param {string} name - Buffer name.
   */
  destroyBuffer(name) {
    const bufferMetadata = this.buffers_.get(name);
    if (bufferMetadata) {
      bufferMetadata.buffer.destroy();
      this.buffers_.delete(name);
      this.emit('bufferDestroyed', { name });
    }
  }

  /**
   * Destroys a pipeline and removes it from management.
   * @param {string} name - Pipeline name.
   */
  destroyPipeline(name) {
    this.pipelines_.delete(name);
    this.emit('pipelineDestroyed', { name });
  }

  /**
   * Destroys a texture and removes it from management.
   * @param {string} name - Texture name.
   */
  destroyTexture(name) {
    const textureMetadata = this.textures_.get(name);
    if (textureMetadata) {
      textureMetadata.texture.destroy();
      this.textures_.delete(name);
      this.emit('textureDestroyed', { name });
    }
  }

  /**
   * Destroys all resources and cleans up the renderer.
   */
  destroy() {
    // Destroy all buffers
    this.buffers_.forEach((metadata, name) => {
      metadata.buffer.destroy();
      this.emit('bufferDestroyed', { name });
    });
    this.buffers_.clear();

    // Destroy all textures
    this.textures_.forEach((metadata, name) => {
      metadata.texture.destroy();
      this.emit('textureDestroyed', { name });
    });
    this.textures_.clear();

    // Destroy render targets
    this.renderTargets_.forEach((metadata, name) => {
      metadata.texture.destroy();
      this.emit('renderTargetDestroyed', { name });
    });
    this.renderTargets_.clear();

    // Clear other collections
    this.pipelines_.clear();
    this.computePipelines_.clear();
    this.samplers_.clear();
    this.bindGroups_.clear();
    this.shaderModules_.clear();

    this.device_ = null;
    this.context_ = null;
    this.isInitialized = false;

    this.emit('destroyed');
  }

  /**
   * Updates buffer data with optimized approach.
   * @param {string} name - Buffer name.
   * @param {ArrayBuffer|ArrayBufferView} data - New data.
   * @param {number} [offset=0] - Offset in bytes.
   * @throws {Error} If buffer not found or renderer not initialized.
   */
  updateBuffer(name, data, offset = 0) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const bufferMetadata = this.buffers_.get(name);
    if (!bufferMetadata) {
      throw new Error(`Buffer "${name}" not found.`);
    }

    const dataToWrite = data instanceof ArrayBuffer ? data : data.buffer;

    // Check if buffer size is sufficient
    if (offset + dataToWrite.byteLength > bufferMetadata.size) {
      console.warn(
        `Buffer "${name}" too small for update. Consider recreating.`
      );
    }

    this.device_.queue.writeBuffer(bufferMetadata.buffer, offset, dataToWrite);

    this.emit('bufferUpdated', { name, offset, size: dataToWrite.byteLength });
  }

  /**
   * Updates texture data with smart resizing.
   * @param {string} name - Texture name.
   * @param {ArrayBufferView} data - Texture data.
   * @param {number} [width] - New texture width (optional).
   * @param {number} [height] - New texture height (optional).
   */
  updateTexture(name, data, width, height) {
    const textureMetadata = this.textures_.get(name);
    if (!textureMetadata) {
      throw new Error(`Texture "${name}" not found.`);
    }

    // Check if resize is needed
    const needsResize =
      width &&
      height &&
      (width !== textureMetadata.width || height !== textureMetadata.height);

    if (needsResize) {
      // Only recreate if size actually changed
      this.destroyTexture(name);
      this.createTexture({
        name,
        width,
        height,
        format: textureMetadata.format,
        usage: textureMetadata.usage,
      });
      // Get updated metadata after recreation
      const newMetadata = this.textures_.get(name);
      textureMetadata.texture = newMetadata.texture;
      textureMetadata.view = newMetadata.view;
      textureMetadata.width = width;
      textureMetadata.height = height;
    }

    const targetWidth = width || textureMetadata.width;
    const targetHeight = height || textureMetadata.height;
    const textureSize = [targetWidth, targetHeight, 1];

    this.device_.queue.writeTexture(
      { texture: textureMetadata.texture },
      data,
      { bytesPerRow: targetWidth * 4 },
      textureSize
    );

    this.emit('textureUpdated', {
      name,
      width: targetWidth,
      height: targetHeight,
    });
  }

  /**
   * Creates a bind group layout.
   * @param {string} name - Layout name.
   * @param {Array<GPUBindGroupLayoutEntry>} entries - Layout entries.
   * @returns {GPUBindGroupLayout} Bind group layout.
   */
  createBindGroupLayout(name, entries) {
    const layout = this.device_.createBindGroupLayout({
      label: name,
      entries,
    });

    this.bindGroupLayouts_.set(name, layout);
    return layout;
  }

  /**
   * Creates a bind group.
   * @param {BindGroupConfig} config - Bind group configuration.
   * @returns {GPUBindGroup} Created bind group.
   * @throws {Error} If renderer not initialized or pipeline not set.
   */
  createBindGroup(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    if (!this.currentPipeline_) {
      throw new Error('No pipeline set. Call setPipeline() first.');
    }

    const pipeline = this.pipelines_.get(this.currentPipeline_);
    const bindGroup = this.device_.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: config.entries,
    });

    this.bindGroups_.set(config.name, bindGroup);
    this.emit('bindGroupCreated', { name: config.name });

    return bindGroup;
  }

  /**
   * Sets the current bind group for rendering.
   * @param {string} bindGroupName - Bind group name.
   * @param {number} [index=0] - Bind group index.
   */
  setBindGroup(bindGroupName, index = 0) {
    const bindGroup = this.bindGroups_.get(bindGroupName);
    if (!bindGroup) {
      throw new Error(`Bind group "${bindGroupName}" not found.`);
    }

    if (!this.currentBindGroups_) {
      this.currentBindGroups_ = {};
    }

    this.currentBindGroups_[index] = bindGroup;
  }

  /**
   * Creates a texture with metadata tracking.
   * @param {TextureConfig} config - Texture configuration.
   * @returns {GPUTexture} Created texture.
   * @throws {Error} If renderer not initialized.
   */
  createTexture(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const format = config.format || 'rgba8unorm';
    const usage =
      config.usage ||
      GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;

    const texture = this.device_.createTexture({
      label: config.name,
      size: [config.width, config.height, 1],
      format,
      usage,
    });

    const view = texture.createView();

    const metadata = {
      texture,
      view,
      width: config.width,
      height: config.height,
      format,
      usage,
    };

    this.textures_.set(config.name, metadata);

    this.emit('textureCreated', {
      name: config.name,
      width: config.width,
      height: config.height,
    });
    return texture;
  }

  /**
   * Creates depth stencil texture with metadata.
   * @private
   */
  createDepthStencilTexture() {
    const format = 'depth24plus-stencil8';
    const usage = GPUTextureUsage.RENDER_ATTACHMENT;

    const texture = this.device_.createTexture({
      size: [this.canvas.width, this.canvas.height, 1],
      format,
      usage,
      sampleCount: this.msaaSampleCount_,
    });

    const view = texture.createView();

    this.depthStencilTexture_ = {
      texture,
      view,
      width: this.canvas.width,
      height: this.canvas.height,
      format,
      usage,
    };
  }

  /**
   * Creates MSAA texture.
   * @private
   */
  createMSAATexture() {
    if (this.msaaSampleCount_ <= 1) {
      return;
    }

    // Destroy existing MSAA texture if it exists
    if (this.msaaTexture_) {
      this.msaaTexture_.texture.destroy();
    }

    const format = this.format_;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT;

    const texture = this.device_.createTexture({
      size: [this.canvas.width, this.canvas.height, 1],
      format,
      usage,
      sampleCount: this.msaaSampleCount_,
    });

    const view = texture.createView();

    this.msaaTexture_ = {
      texture,
      view,
      width: this.canvas.width,
      height: this.canvas.height,
      format,
      usage,
      sampleCount: this.msaaSampleCount_,
    };
  }

  /**
   * Creates a sampler.
   * @param {SamplerConfig} config - Sampler configuration.
   * @returns {GPUSampler} Created sampler.
   * @throws {Error} If renderer not initialized.
   */
  createSampler(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const sampler = this.device_.createSampler({
      label: config.name,
      addressModeU: config.addressModeU || 'clamp-to-edge',
      addressModeV: config.addressModeV || 'clamp-to-edge',
      addressModeW: config.addressModeW || 'clamp-to-edge',
      magFilter: config.magFilter || 'linear',
      minFilter: config.minFilter || 'linear',
    });

    this.samplers_.set(config.name, sampler);
    this.emit('samplerCreated', { name: config.name });

    return sampler;
  }

  /**
   * Creates a GPU buffer with enhanced metadata tracking.
   * @param {BufferConfig} config - Buffer configuration.
   * @returns {GPUBuffer} Created buffer.
   * @throws {Error} If renderer not initialized or config invalid.
   */
  createBuffer(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized. Call initialize() first.');
    }

    if (!config.name) {
      throw new Error('Buffer name is required.');
    }

    if (!config.data) {
      throw new Error('Buffer data is required.');
    }

    if (!config.usage) {
      throw new Error('Buffer usage flags are required.');
    }

    const buffer = this.device_.createBuffer({
      label: config.name,
      size: config.data.byteLength,
      usage: config.usage,
      mappedAtCreation: true,
    });

    const arrayBuffer =
      config.data instanceof ArrayBuffer
        ? new Uint8Array(buffer.getMappedRange())
        : new config.data.constructor(buffer.getMappedRange());

    arrayBuffer.set(
      config.data instanceof ArrayBuffer
        ? new Uint8Array(config.data)
        : config.data
    );

    buffer.unmap();

    // Store metadata including explicit index format
    const metadata = {
      buffer,
      size: config.data.byteLength,
      usage: config.usage,
      indexFormat: config.indexFormat, // Explicitly store format instead of guessing
    };

    this.buffers_.set(config.name, metadata);

    this.emit('bufferCreated', {
      name: config.name,
      size: config.data.byteLength,
      usage: config.usage,
    });

    return buffer;
  }

  /**
   * Creates a render target with metadata tracking.
   * @param {RenderTargetConfig} config - Render target configuration.
   * @returns {GPUTexture} Created render target.
   * @throws {Error} If renderer not initialized.
   */
  createRenderTarget(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const format = config.format || 'rgba8unorm';
    const usage =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

    const texture = this.device_.createTexture({
      label: config.name,
      size: [config.width, config.height, 1],
      format,
      usage,
      sampleCount: config.sampleCount || 1,
    });

    const view = texture.createView();

    const metadata = {
      texture,
      view,
      width: config.width,
      height: config.height,
      format,
      usage,
      sampleCount: config.sampleCount || 1,
    };

    this.renderTargets_.set(config.name, metadata);

    this.emit('renderTargetCreated', {
      name: config.name,
      width: config.width,
      height: config.height,
    });

    return texture;
  }

  /**
   * Sets the current render target.
   * @param {?string} name - Render target name or null for default canvas.
   */
  setRenderTarget(name) {
    if (name && !this.renderTargets_.has(name)) {
      throw new Error(`Render target "${name}" not found.`);
    }
    this.currentRenderTarget_ = name;
  }

  /**
   * Sets blend state.
   * @param {boolean} enabled - Whether blending is enabled.
   * @param {GPUBlendState} [config] - Blend configuration.
   */
  setBlendState(enabled, config = {}) {
    this.blendEnabled_ = enabled;
    this.blendState_ = {
      color: {
        srcFactor: config.srcFactor || 'src-alpha',
        dstFactor: config.dstFactor || 'one-minus-src-alpha',
        operation: config.colorOperation || 'add',
      },
      alpha: {
        srcFactor: config.alphaSrcFactor || 'one',
        dstFactor: config.alphaDstFactor || 'one-minus-src-alpha',
        operation: config.alphaOperation || 'add',
      },
    };
  }

  /**
   * Sets depth state.
   * @param {boolean} enabled - Whether depth testing is enabled.
   * @param {Object} [config] - Depth stencil configuration.
   * @param {GPUCompareFunction} [config.compare='less'] - Depth compare function.
   * @param {boolean} [config.depthWriteEnabled=true] - Whether depth writing is enabled.
   */
  setDepthState(enabled, config = {}) {
    this.depthEnabled_ = enabled;
    this.depthStencilState_ = {
      depthWriteEnabled: config.depthWriteEnabled !== false,
      depthCompare: config.compare || 'less',
      format: 'depth24plus-stencil8',
    };

    // Create depth texture if needed
    if (enabled && !this.depthStencilTexture_) {
      this.createDepthStencilTexture();
    }
  }

  /**
   * Sets cull mode.
   * @param {GPUCullMode} mode - Cull mode.
   */
  setCullMode(mode) {
    this.cullMode_ = mode;
  }

  /**
   * Gets or creates a shader module.
   * @param {string} code - Shader code.
   * @returns {GPUShaderModule} Shader module.
   * @private
   */
  getShaderModule_(code) {
    if (!this.shaderModules_.has(code)) {
      const module = this.device_.createShaderModule({
        code,
        label: `Cached shader module`,
      });
      this.shaderModules_.set(code, module);
    }
    return this.shaderModules_.get(code);
  }

  /**
   * Registers an event listener.
   * @param {string} event - Event name.
   * @param {EventCallback} callback - Callback function.
   */
  on(event, callback) {
    if (!this.eventListeners_.has(event)) {
      this.eventListeners_.set(event, new Set());
    }
    this.eventListeners_.get(event).add(callback);
  }

  /**
   * Removes an event listener.
   * @param {string} event - Event name.
   * @param {EventCallback} callback - Callback function.
   */
  off(event, callback) {
    this.eventListeners_.get(event)?.delete(callback);
  }

  /**
   * Emits an event.
   * @param {string} event - Event name.
   * @param {*} [data] - Event data.
   * @private
   */
  emit(event, data) {
    this.eventListeners_.get(event)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in event listener for ${event}:`, error);
      }
    });
  }

  /**
   * Begins a new frame for statistics.
   */
  beginFrame() {
    this.frameCount_++;
  }

  /**
   * Gets renderer statistics.
   * @returns {Object} Statistics object.
   */
  getStats() {
    const now = performance.now();
    const elapsed = (now - this.startTime_) / 1000;
    const fps = this.frameCount_ / elapsed;

    return {
      frameCount: this.frameCount_,
      fps: fps,
      buffers: this.buffers_.size,
      textures: this.textures_.size,
      pipelines: this.pipelines_.size,
      renderTargets: this.renderTargets_.size,
      bindGroups: this.bindGroups_.size,
      samplers: this.samplers_.size,
    };
  }

  /**
   * Sets vertex buffer for rendering.
   * @param {number} slot - Vertex buffer slot.
   * @param {string} bufferName - Buffer name.
   * @param {number} [offset=0] - Buffer offset.
   */
  setVertexBuffer(slot, bufferName, offset = 0) {
    if (!this.currentVertexBuffers_) {
      this.currentVertexBuffers_ = {};
    }

    const bufferMetadata = this.buffers_.get(bufferName);
    if (bufferMetadata) {
      this.currentVertexBuffers_[slot] = {
        buffer: bufferMetadata.buffer,
        offset,
      };
    } else {
      console.warn(`Vertex buffer "${bufferName}" not found.`);
    }
  }

  /**
   * Creates a compute pipeline.
   * @param {string} name - Pipeline name.
   * @param {string} computeShader - Compute shader code.
   * @returns {GPUComputePipeline} Compute pipeline.
   * @throws {Error} If renderer not initialized.
   */
  createComputePipeline(name, computeShader) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const shaderModule = this.getShaderModule_(computeShader);
    const pipeline = this.device_.createComputePipeline({
      label: name,
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    this.computePipelines_.set(name, pipeline);
    this.emit('computePipelineCreated', { name });

    return pipeline;
  }

  /**
   * Dispatches a compute shader.
   * @param {string} pipelineName - Compute pipeline name.
   * @param {Object} config - Dispatch configuration.
   * @param {number} config.workgroupCountX - X workgroup count.
   * @param {number} [config.workgroupCountY=1] - Y workgroup count.
   * @param {number} [config.workgroupCountZ=1] - Z workgroup count.
   * @param {Object} [config.bindGroups] - Bind groups to set.
   */
  dispatchCompute(pipelineName, config) {
    const pipeline = this.computePipelines_.get(pipelineName);
    if (!pipeline) {
      throw new Error(`Compute pipeline "${pipelineName}" not found.`);
    }

    const encoder = this.device_.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);

    // Set bind groups if provided
    if (config.bindGroups) {
      Object.entries(config.bindGroups).forEach(([index, bindGroupName]) => {
        const bindGroup = this.bindGroups_.get(bindGroupName);
        if (bindGroup) {
          pass.setBindGroup(parseInt(index), bindGroup);
        }
      });
    }

    pass.dispatchWorkgroups(
      config.workgroupCountX,
      config.workgroupCountY || 1,
      config.workgroupCountZ || 1
    );

    pass.end();
    this.device_.queue.submit([encoder.finish()]);

    this.emit('computeDispatched', {
      pipelineName,
      workgroupCountX: config.workgroupCountX,
      workgroupCountY: config.workgroupCountY,
      workgroupCountZ: config.workgroupCountZ,
    });
  }

  /**
   * Unified pipeline creation method that handles both auto and explicit layouts.
   * This replaces both createPipeline and createPipelineWithLayout to eliminate duplication.
   * @param {string} name - Pipeline name.
   * @param {PipelineConfig} config - Pipeline configuration.
   * @returns {GPURenderPipeline} Created pipeline.
   * @throws {Error} If renderer is not initialized.
   */
  createPipeline(name, config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized. Call initialize() first.');
    }

    const shaderCode = `${config.shaders.vertex}\n${config.shaders.fragment}`;
    const shaderModule = this.getShaderModule_(shaderCode);

    // Determine layout: explicit or auto
    const layout = config.bindGroupLayouts
      ? this.device_.createPipelineLayout({
          bindGroupLayouts: config.bindGroupLayouts,
        })
      : 'auto';

    const pipelineDescriptor = {
      label: `${name} pipeline`,
      layout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: config.vertexBuffers,
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: this.format_,
            blend: this.blendEnabled_ ? this.blendState_ : undefined,
          },
        ],
      },
      primitive: {
        topology: config.primitive?.topology || 'triangle-list',
        cullMode: this.cullMode_,
        ...config.primitive,
      },
      depthStencil: this.depthEnabled_
        ? config.depthStencil || this.depthStencilState_
        : undefined,
      multisample: {
        count: this.msaaSampleCount_,
      },
    };

    const pipeline = this.device_.createRenderPipeline(pipelineDescriptor);
    this.pipelines_.set(name, pipeline);

    this.emit('pipelineCreated', {
      name,
      config,
      hasExplicitLayout: !!config.bindGroupLayouts,
    });
    return pipeline;
  }

  /**
   * Sets the current clearColor.
   * @override
   * @param {number} r - Red cleaning color component (0-1).
   * @param {number} g - Green cleaning color component (0-1).
   * @param {number} b - Blue cleaning color component (0-1).
   * @param {number} a - Alpha cleaning color component (0-1).
   */
  setClearColor(r, g, b, a) {
    this.clearColor_ = { r, g, b, a };
  }

  /**
   * Sets the current pipeline for rendering.
   * @param {string} name - Pipeline name.
   */
  setPipeline(name) {
    if (this.pipelines_.has(name)) {
      this.currentPipeline_ = name;
      this.emit('pipelineSet', { name });
    }
  }

  /**
   * Enables or disables MSAA.
   * @param {boolean} enabled - If true then enable MSAA, else disable.
   * @returns {boolean} True if MSAA was successfully enabled, false otherwise.
   */
  setMSAA(enabled) {
    if (!this.isInitialized) {
      console.warn('Renderer not initialized. Call initialize() first.');
      return false;
    }

    this.msaaSampleCount_ = enabled ? 4 : 1;

    // Recreate MSAA texture if needed (sample count > 1)
    if (this.msaaSampleCount_ > 1) {
      this.createMSAATexture();
    } else if (this.msaaTexture_) {
      // Clean up MSAA texture if disabling MSAA
      this.msaaTexture_.texture.destroy();
      this.msaaTexture_ = null;
    }

    this.emit('msaaChanged', { samples: this.msaaSampleCount_ });
    return true;
  }

  /**
   * Render method.
   * @param {Object} drawCall - Draw call configuration.
   */
  render(drawCall = {}) {
    this.beginFrame();

    if (!this.isInitialized) {
      console.warn('Renderer not initialized. Call initialize() first.');
      return;
    }

    if (!this.currentPipeline_) {
      throw new Error('No pipeline set. Call setPipeline() first.');
    }

    const encoder = this.device_.createCommandEncoder({
      label: `Frame ${this.frameCount_} command encoder`,
    });

    const renderPassDescriptor = {
      label: 'Render pass',
      colorAttachments: [
        {
          view: this.currentRenderTarget_
            ? this.renderTargets_.get(this.currentRenderTarget_).view
            : this.msaaSampleCount_ > 1
            ? this.msaaTexture_.view
            : this.context_.getCurrentTexture().createView(),
          resolveTarget:
            this.msaaSampleCount_ > 1 && !this.currentRenderTarget_
              ? this.context_.getCurrentTexture().createView()
              : undefined,
          clearValue: this.clearColor_,
          loadOp: 'clear',
          storeOp: this.msaaSampleCount_ > 1 ? 'discard' : 'store',
        },
      ],
    };

    if (this.currentRenderTarget_ && this.msaaSampleCount_ > 1) {
      const rtMetadata = this.renderTargets_.get(this.currentRenderTarget_);
      if (rtMetadata.sampleCount > 1) {
        // Render target has its own MSAA, no resolve needed here
        renderPassDescriptor.colorAttachments[0].resolveTarget = undefined;
      }
    }

    // Add depth/stencil attachment if enabled
    if (this.depthEnabled_ && this.depthStencilTexture_) {
      renderPassDescriptor.depthStencilAttachment = {
        view: this.depthStencilTexture_.view,
        depthClearValue: this.clearDepth_,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        stencilClearValue: this.clearStencil_,
        stencilLoadOp: 'clear',
        stencilStoreOp: 'store',
      };
    }

    const pass = encoder.beginRenderPass(renderPassDescriptor);
    const pipeline = this.pipelines_.get(this.currentPipeline_);

    pass.setPipeline(pipeline);

    // Set vertex buffers
    if (drawCall.vertexBuffers) {
      Object.entries(drawCall.vertexBuffers).forEach(([slot, bufferName]) => {
        const bufferMetadata = this.buffers_.get(bufferName);
        if (bufferMetadata) {
          pass.setVertexBuffer(parseInt(slot), bufferMetadata.buffer);
        }
      });
    }

    // Set bind groups
    if (drawCall.bindGroups) {
      Object.entries(drawCall.bindGroups).forEach(([index, bindGroupName]) => {
        const bindGroup = this.bindGroups_.get(bindGroupName);
        if (bindGroup) {
          pass.setBindGroup(parseInt(index), bindGroup);
        }
      });
    }

    // Draw call
    if (drawCall.indexBuffer && this.buffers_.has(drawCall.indexBuffer)) {
      const bufferMetadata = this.buffers_.get(drawCall.indexBuffer);
      pass.setIndexBuffer(
        bufferMetadata.buffer,
        bufferMetadata.indexFormat || 'uint16' // Use stored format or default
      );
      pass.drawIndexed(
        drawCall.indexCount || 0,
        drawCall.instanceCount || 1,
        drawCall.firstIndex || 0,
        drawCall.baseVertex || 0,
        drawCall.firstInstance || 0
      );
    } else if (drawCall.vertexCount) {
      pass.draw(
        drawCall.vertexCount,
        drawCall.instanceCount || 1,
        drawCall.firstVertex || 0,
        drawCall.firstInstance || 0
      );
    }

    pass.end();
    this.device_.queue.submit([encoder.finish()]);

    this.emit('frameRendered', { frameCount: this.frameCount_, drawCall });
  }

  /**
   * Sets the size of the rendering viewport.
   * @override
   * @param {number} width - Width in pixels.
   * @param {number} height - Height in pixels.
   */
  setSize(width, height) {
    if (!this.isInitialized) {
      console.warn('Renderer not initialized. Call initialize() first.');
      return;
    }

    const actualWidth = Math.max(1, width);
    const actualHeight = Math.max(1, height);

    // Reconfigure WebGPU context with new size
    if (this.context_) {
      this.context_.configure({
        device: this.device_,
        format: this.format_,
        alphaMode: 'premultiplied',
        size: [actualWidth, actualHeight],
      });
    }

    // Recreate MSAA texture if it exists
    if (this.msaaSampleCount_ > 1) {
      this.createMSAATexture();
    }

    // Recreate depth texture if it exists
    if (this.depthStencilTexture_) {
      this.depthStencilTexture_.texture.destroy();
      this.createDepthStencilTexture();
    }

    this.emit('resize', { width: actualWidth, height: actualHeight });
  }

  /**
   * Checks if WebGPU is supported in the current environment.
   * @override
   * @returns {boolean} True if WebGPU is supported, otherwise false.
   */
  static isSupported() {
    return (
      typeof navigator !== 'undefined' &&
      navigator.gpu !== undefined &&
      typeof GPUAdapter !== 'undefined' &&
      typeof GPUDevice !== 'undefined'
    );
  }

  // Enhanced getters that return metadata or buffers as appropriate
  getBuffer(name) {
    const metadata = this.buffers_.get(name);
    return metadata ? metadata.buffer : null;
  }

  getBufferMetadata(name) {
    return this.buffers_.get(name) || null;
  }

  getPipeline(name) {
    return this.pipelines_.get(name) || null;
  }

  getTexture(name) {
    const metadata = this.textures_.get(name);
    return metadata ? metadata.texture : null;
  }

  getTextureView(name) {
    const metadata = this.textures_.get(name);
    return metadata ? metadata.view : null;
  }

  getTextureMetadata(name) {
    return this.textures_.get(name) || null;
  }

  getSampler(name) {
    return this.samplers_.get(name) || null;
  }

  getMSAASampleCount() {
    return this.msaaSampleCount_;
  }

  getBindGroup(name) {
    return this.bindGroups_.get(name) || null;
  }

  getRenderTarget(name) {
    const metadata = this.renderTargets_.get(name);
    return metadata ? metadata.texture : null;
  }

  getRenderTargetView(name) {
    const metadata = this.renderTargets_.get(name);
    return metadata ? metadata.view : null;
  }

  getRenderTargetMetadata(name) {
    return this.renderTargets_.get(name) || null;
  }

  getIndexFormat(bufferName) {
    const metadata = this.buffers_.get(bufferName);
    return metadata?.indexFormat || 'uint16';
  }

  /**
   * Helper method to create index buffers with explicit format.
   * @param {string} name - Buffer name.
   * @param {Uint16Array|Uint32Array} indices - Index data.
   * @returns {GPUBuffer} Created index buffer.
   */
  createIndexBuffer(name, indices) {
    const indexFormat = indices instanceof Uint32Array ? 'uint32' : 'uint16';

    return this.createBuffer({
      name,
      data: indices,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      indexFormat, // Explicitly store the format
    });
  }

  /**
   * Helper method to create vertex buffers.
   * @param {string} name - Buffer name.
   * @param {Float32Array} vertices - Vertex data.
   * @returns {GPUBuffer} Created vertex buffer.
   */
  createVertexBuffer(name, vertices) {
    return this.createBuffer({
      name,
      data: vertices,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Helper method to create uniform buffers.
   * @param {string} name - Buffer name.
   * @param {ArrayBufferView} data - Uniform data.
   * @returns {GPUBuffer} Created uniform buffer.
   */
  createUniformBuffer(name, data) {
    return this.createBuffer({
      name,
      data,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
}

/** Returns HTML template for the error message
 *
 * Insert into the DOM when catched the error
 *
 * @param {string} error - Catched error
 * @returns {string}
 */
export function HTMLWGPUErrorMessage(error) {
  return `
    <div style="color: white; font-family: Arial; text-align: center; padding: 50px;">
      <h2>WebGPU Error</h2>
      <p>Probably rendering code is broken.</p>
      <p style="color: #ff6b6b; margin-top: 20px;">Error info for developers: ${error.message}</p>
      <div style="margin-top: 30px;">
        <a href="https://developer.chrome.com/docs/web-platform/webgpu"
           target="_blank"
           style="color: #4ecdc4; text-decoration: none;">
           Learn more about WebGPU
        </a>
      </div>
    </div>
  `;
}

/** Returns HTML template for the error message
 *
 * Insert into the DOM when isSupported() failed.
 *
 * @param {string} error - Catched error
 * @returns {string}
 */
export function HTMLWGPUUnsupportedMessage(msg) {
  return `
    <div style="color: white; font-family: Arial; text-align: center; padding: 50px;">
      <h2>WebGPU Not Supported</h2>
      <p>Your browser does not support WebGPU or it's disabled.</p>
      <p>Please use Chrome 113+ or Firefox 141+ with WebGPU enabled.</p>
      <p style="color: #6d6d6dff; margin-top: 20px;">Developer's note: ${msg}</p>
      <div style="margin-top: 30px;">
        <a href="https://developer.chrome.com/docs/web-platform/webgpu"
          target="_blank"
          style="color: #4ecdc4; text-decoration: none;">
          Learn more about WebGPU
        </a>
      </div>
    </div>
  `;
}
