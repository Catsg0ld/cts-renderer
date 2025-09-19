/**
 * @fileoverview WebGL API based renderer implementation.
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
 * @property {number} usage - Buffer usage flags (contextSTATIC_DRAW, contextDYNAMIC_DRAW, etc.).
 * @property {number} [type] - Buffer type (contextARRAY_BUFFER, contextELEMENT_ARRAY_BUFFER).
 */

/**
 * Program configuration object.
 * @typedef {Object} ProgramConfig
 * @property {ShaderConfig} shaders - Shader configuration.
 * @property {Object} attributes - Vertex attributes configuration.
 * @property {Object} uniforms - Uniforms configuration.
 */

/**
 * Texture configuration object.
 * @typedef {Object} TextureConfig
 * @property {string} name - Texture name.
 * @property {number} width - Texture width.
 * @property {number} height - Texture height.
 * @property {ArrayBufferView} data - Texture data.
 * @property {number} [format=contextRGBA] - Texture format.
 * @property {number} [internalFormat=contextRGBA] - Texture internal format.
 * @property {number} [type=contextUNSIGNED_BYTE] - Texture data type.
 * @property {Object} [parameters] - Texture parameters.
 */

/**
 * Vertex attribute configuration.
 * @typedef {Object} AttributeConfig
 * @property {number} size - Number of components per attribute.
 * @property {number} type - Data type (contextFLOAT, etc.).
 * @property {boolean} normalized - Whether data is normalized.
 * @property {number} stride - Byte stride between consecutive attributes.
 * @property {number} offset - Offset in bytes.
 */

/**
 * Event listener callback.
 * @callback EventCallback
 * @param {*} data - Event data.
 */

/**
 * Buffer metadata for resource management.
 * @typedef {Object} BufferMetadata
 * @property {WebGLBuffer} buffer - The WebGL buffer.
 * @property {number} size - Buffer size in bytes.
 * @property {number} usage - Buffer usage flags.
 * @property {number} type - Buffer type.
 */

/**
 * Texture metadata for resource management.
 * @typedef {Object} TextureMetadata
 * @property {WebGLTexture} texture - The WebGL texture.
 * @property {number} width - Texture width.
 * @property {number} height - Texture height.
 * @property {number} format - Texture format.
 * @property {number} internalFormat - Texture internal format.
 */

/**
 * Program metadata for resource management.
 * @typedef {Object} ProgramMetadata
 * @property {WebGLProgram} program - The WebGL program.
 * @property {WebGLShader} vertexShader - Vertex shader.
 * @property {WebGLShader} fragmentShader - Fragment shader.
 * @property {Object} attributes - Attribute locations.
 * @property {Object} uniforms - Uniform locations.
 */

/**
 * Renderer using WebGL1 API for graphics rendering.
 * @extends Renderer
 */
export class WGLRenderer extends Renderer {
  /**
   * Creates an instance of WGLRenderer.
   * @param {HTMLCanvasElement} canvas - Canvas element for rendering.
   * @param {Object} [options] - Renderer options.
   */
  constructor(canvas, options = {}) {
    super(canvas);

    /**
     * WebGL context.
     * @type {?WebGLRenderingContext}
     * @private
     */
    this.context = null;

    /**
     * Renderer options.
     * @type {Object}
     * @private
     */
    this.options_ = {
      alpha: options.alpha !== false,
      depth: options.depth !== false,
      stencil: options.stencil || false,
      antialias: options.antialias !== false,
      premultipliedAlpha: options.premultipliedAlpha || false,
      preserveDrawingBuffer: options.preserveDrawingBuffer || false,
      ...options,
    };

    /**
     * Shader programs.
     * @type {Map<string, ProgramMetadata>}
     * @private
     */
    this.programs_ = new Map();

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
     * Framebuffers.
     * @type {Map<string, WebGLFramebuffer>}
     * @private
     */
    this.framebuffers_ = new Map();

    /**
     * Renderbuffers.
     * @type {Map<string, WebGLRenderbuffer>}
     * @private
     */
    this.renderbuffers_ = new Map();

    /**
     * Current program name.
     * @type {?string}
     * @private
     */
    this.currentProgram_ = null;

    /**
     * Current clear color.
     * @type {number[]}
     * @private
     */
    this.clearColor_ = [0.1, 0.1, 0.1, 1.0];

    /**
     * Event listeners.
     * @type {Map<string, Set<EventCallback>>}
     * @private
     */
    this.eventListeners_ = new Map();

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
     * Current framebuffer.
     * @type {?string}
     * @private
     */
    this.currentFramebuffer_ = null;

    /**
     * Blend state enabled.
     * @type {boolean}
     * @private
     */
    this.blendEnabled_ = false;

    /**
     * Depth test state enabled.
     * @type {boolean}
     * @private
     */
    this.depthTestEnabled_ = false;

    /**
     * Cull face state enabled.
     * @type {boolean}
     * @private
     */
    this.cullFaceEnabled_ = false;

    /**
     * Current cull face mode.
     * @type {number}
     * @private
     */
    this.cullFaceMode_ = 0x0405; // GL.BACK

    /**
     * Current depth function.
     * @type {number}
     * @private
     */
    this.depthFunc_ = 0x0201; // GL.LESS

    /**
     * Current blend function.
     * @type {Object}
     * @private
     */
    this.blendFunc_ = {
      src: 0x0302, // GL.SRC_ALPHA
      dst: 0x0303, // GL.ONE_MINUS_SRC_ALPHA
    };
  }

  /**
   * Initializes WebGL renderer.
   * @override
   * @returns {Promise<boolean>} Promise that resolves to true on successful initialization.
   * @throws {Error} If WebGL is not supported or initialization fails.
   */
  async initialize() {
    if (!WGLRenderer.isSupported()) {
      throw new Error('WebGL is not supported in this browser.');
    }

    try {
      // Get WebGL context
      this.context =
        this.canvas.getContext('webgl', this.options_) ||
        this.canvas.getContext('experimental-webgl', this.options_);

      if (!this.context) {
        throw new Error('Failed to get WebGL context.');
      }

      // Set initial state
      this.setClearColor(...this.clearColor_);
      this.setDepthTest(this.depthTestEnabled_);
      this.setBlendState(this.blendEnabled_);
      this.setCullFace(this.cullFaceEnabled_, this.cullFaceMode_);

      this.isInitialized = true;
      this.emit('initialized', { gl: this.context, options: this.options_ });
      return true;
    } catch (error) {
      console.error('Failed to initialize WGLRenderer:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Compiles a shader.
   * @param {number} type - Shader type (GL.VERTEX_SHADER or GL.FRAGMENT_SHADER).
   * @param {string} source - Shader source code.
   * @returns {WebGLShader} Compiled shader.
   * @private
   */
  compileShader_(type, source) {
    const shader = this.context.createShader(type);
    this.context.shaderSource(shader, source);
    this.context.compileShader(shader);

    if (!this.context.getShaderParameter(shader, this.context.COMPILE_STATUS)) {
      const info = this.context.getShaderInfoLog(shader);
      this.context.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${info}`);
    }

    return shader;
  }

  /**
   * Creates a shader program.
   * @param {string} name - Program name.
   * @param {ShaderConfig} shaders - Shader configuration.
   * @returns {WebGLProgram} Created program.
   * @throws {Error} If renderer not initialized.
   */
  createProgram(name, shaders) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    try {
      const vertexShader = this.compileShader_(
        this.context.VERTEX_SHADER,
        shaders.vertex
      );
      const fragmentShader = this.compileShader_(
        this.context.FRAGMENT_SHADER,
        shaders.fragment
      );

      const program = this.context.createProgram();
      this.context.attachShader(program, vertexShader);
      this.context.attachShader(program, fragmentShader);
      this.context.linkProgram(program);

      if (
        !this.context.getProgramParameter(program, this.context.LINK_STATUS)
      ) {
        const info = this.context.getProgramInfoLog(program);
        throw new Error(`Program linking failed: ${info}`);
      }

      // Get attribute and uniform locations
      const attributes = {};
      const uniforms = {};

      // Store program metadata
      const metadata = {
        program,
        vertexShader,
        fragmentShader,
        attributes,
        uniforms,
      };

      this.programs_.set(name, metadata);

      this.emit('programCreated', { name });
      return program;
    } catch (error) {
      console.error(`Failed to create program "${name}":`, error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Sets up vertex attributes for a program.
   * @param {string} programName - Program name.
   * @param {Object} attributesConfig - Attributes configuration.
   */
  setupVertexAttributes(programName, attributesConfig) {
    const programMetadata = this.programs_.get(programName);
    if (!programMetadata) {
      throw new Error(`Program "${programName}" not found.`);
    }

    this.useProgram(programName);

    Object.entries(attributesConfig).forEach(([name, config]) => {
      const location = this.context.getAttribLocation(
        programMetadata.program,
        name
      );
      if (location === -1) {
        console.warn(
          `Attribute "${name}" not found in program "${programName}"`
        );
        return;
      }

      programMetadata.attributes[name] = {
        location,
        size: config.size || 3,
        type: config.type || this.context.FLOAT,
        normalized: config.normalized || false,
        stride: config.stride || 0,
        offset: config.offset || 0,
      };
    });

    this.emit('vertexAttributesSetup', {
      programName,
      attributes: attributesConfig,
    });
  }

  /**
   * Sets uniform value.
   * @param {string} programName - Program name.
   * @param {string} uniformName - Uniform name.
   * @param {*} value - Uniform value.
   */
  setUniform(programName, uniformName, value) {
    const programMetadata = this.programs_.get(programName);
    if (!programMetadata) {
      throw new Error(`Program "${programName}" not found.`);
    }

    this.useProgram(programName);

    let location = programMetadata.uniforms[uniformName];
    if (location === undefined) {
      location = this.context.getUniformLocation(
        programMetadata.program,
        uniformName
      );
      if (!location) {
        console.warn(
          `Uniform "${uniformName}" not found in program "${programName}"`
        );
        return;
      }
      programMetadata.uniforms[uniformName] = location;
    }

    if (value instanceof Float32Array) {
      switch (value.length) {
        case 4:
          this.context.uniform4fv(location, value);
          break;
        case 3:
          this.context.uniform3fv(location, value);
          break;
        case 2:
          this.context.uniform2fv(location, value);
          break;
        case 1:
          this.context.uniform1fv(location, value);
          break;
        case 16:
          this.context.uniformMatrix4fv(location, false, value);
          break;
        case 9:
          this.context.uniformMatrix3fv(location, false, value);
          break;
        default:
          this.context.uniform1fv(location, value);
      }
    } else if (typeof value === 'number') {
      this.context.uniform1f(location, value);
    } else if (Array.isArray(value)) {
      switch (value.length) {
        case 4:
          this.context.uniform4f(location, ...value);
          break;
        case 3:
          this.context.uniform3f(location, ...value);
          break;
        case 2:
          this.context.uniform2f(location, ...value);
          break;
        case 1:
          this.context.uniform1f(location, value[0]);
          break;
        default:
          this.context.uniform1fv(location, new Float32Array(value));
      }
    }
  }

  /**
   * Creates a GPU buffer.
   * @param {BufferConfig} config - Buffer configuration.
   * @returns {WebGLBuffer} Created buffer.
   * @throws {Error} If renderer not initialized.
   */
  createBuffer(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const buffer = this.context.createBuffer();
    if (!buffer) {
      throw new Error('Failed to create buffer.');
    }

    const type = config.type || this.context.ARRAY_BUFFER;
    const usage = config.usage || this.context.STATIC_DRAW;

    this.context.bindBuffer(type, buffer);
    this.context.bufferData(type, config.data, usage);

    const metadata = {
      buffer,
      size: config.data.byteLength,
      usage,
      type,
    };

    this.buffers_.set(config.name, metadata);
    this.context.bindBuffer(type, null);

    this.emit('bufferCreated', {
      name: config.name,
      size: config.data.byteLength,
      usage,
      type,
    });

    return buffer;
  }

  /**
   * Updates buffer data.
   * @param {string} name - Buffer name.
   * @param {ArrayBuffer|ArrayBufferView} data - New data.
   * @param {number} [offset=0] - Offset in bytes.
   */
  updateBuffer(name, data, offset = 0) {
    const bufferMetadata = this.buffers_.get(name);
    if (!bufferMetadata) {
      throw new Error(`Buffer "${name}" not found.`);
    }

    this.context.bindBuffer(bufferMetadata.type, bufferMetadata.buffer);

    if (offset === 0 && data.byteLength === bufferMetadata.size) {
      this.context.bufferData(bufferMetadata.type, data, bufferMetadata.usage);
    } else {
      this.context.bufferSubData(bufferMetadata.type, offset, data);
    }

    this.context.bindBuffer(bufferMetadata.type, null);

    this.emit('bufferUpdated', { name, offset, size: data.byteLength });
  }

  /**
   * Creates a texture.
   * @param {TextureConfig} config - Texture configuration.
   * @returns {WebGLTexture} Created texture.
   * @throws {Error} If renderer not initialized.
   */
  createTexture(config) {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized.');
    }

    const texture = this.context.createTexture();
    if (!texture) {
      throw new Error('Failed to create texture.');
    }

    const format = config.format || this.context.RGBA;
    const internalFormat = config.internalFormat || this.context.RGBA;
    const type = config.type || this.context.UNSIGNED_BYTE;

    this.context.bindTexture(this.context.TEXTURE_2D, texture);

    // Set texture parameters
    if (config.parameters) {
      Object.entries(config.parameters).forEach(([pname, pvalue]) => {
        const param = this.context[pname];
        if (param !== undefined) {
          this.context.texParameteri(this.context.TEXTURE_2D, param, pvalue);
        }
      });
    } else {
      // Default parameters
      this.context.texParameteri(
        this.context.TEXTURE_2D,
        this.context.TEXTURE_MIN_FILTER,
        this.context.LINEAR
      );
      this.context.texParameteri(
        this.context.TEXTURE_2D,
        this.context.TEXTURE_MAG_FILTER,
        this.context.LINEAR
      );
      this.context.texParameteri(
        this.context.TEXTURE_2D,
        this.context.TEXTURE_WRAP_S,
        this.context.CLAMP_TO_EDGE
      );
      this.context.texParameteri(
        this.context.TEXTURE_2D,
        this.context.TEXTURE_WRAP_T,
        this.context.CLAMP_TO_EDGE
      );
    }

    this.context.texImage2D(
      this.context.TEXTURE_2D,
      0,
      internalFormat,
      config.width,
      config.height,
      0,
      format,
      type,
      config.data
    );

    const metadata = {
      texture,
      width: config.width,
      height: config.height,
      format,
      internalFormat,
    };

    this.textures_.set(config.name, metadata);
    this.context.bindTexture(this.context.TEXTURE_2D, null);

    this.emit('textureCreated', {
      name: config.name,
      width: config.width,
      height: config.height,
    });

    return texture;
  }

  /**
   * Creates a framebuffer.
   * @param {string} name - Framebuffer name.
   * @param {Object} attachments - Framebuffer attachments.
   * @returns {WebGLFramebuffer} Created framebuffer.
   */
  createFramebuffer(name, attachments) {
    const framebuffer = this.context.createFramebuffer();
    if (!framebuffer) {
      throw new Error('Failed to create framebuffer.');
    }

    this.context.bindFramebuffer(this.context.FRAMEBUFFER, framebuffer);

    Object.entries(attachments).forEach(([attachmentPoint, attachmentName]) => {
      const textureMetadata = this.textures_.get(attachmentName);
      if (textureMetadata) {
        const attachment = this.context[attachmentPoint];
        this.context.framebufferTexture2D(
          this.context.FRAMEBUFFER,
          attachment,
          this.context.TEXTURE_2D,
          textureMetadata.texture,
          0
        );
      }
    });

    // Check framebuffer status
    const status = this.context.checkFramebufferStatus(
      this.context.FRAMEBUFFER
    );
    if (status !== this.context.FRAMEBUFFER_COMPLETE) {
      this.context.bindFramebuffer(this.context.FRAMEBUFFER, null);
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    this.framebuffers_.set(name, framebuffer);
    this.context.bindFramebuffer(this.context.FRAMEBUFFER, null);

    this.emit('framebufferCreated', { name });
    return framebuffer;
  }

  /**
   * Sets the current clear color.
   * @override
   * @param {number} r - Red component (0-1).
   * @param {number} g - Green component (0-1).
   * @param {number} b - Blue component (0-1).
   * @param {number} a - Alpha component (0-1).
   */
  setClearColor(r, g, b, a) {
    this.clearColor_ = [r, g, b, a];
    this.context.clearColor(r, g, b, a);
  }

  /**
   * Clears the rendering buffers.
   * @override
   * @param {number} r - Red component of clear color (0-1).
   * @param {number} g - Green component of clear color (0-1).
   * @param {number} b - Blue component of clear color (0-1).
   * @param {number} a - Alpha component of clear color (0-1).
   */
  clear(r, g, b, a) {
    this.setClearColor(r, g, b, a);
    let mask = this.context.COLOR_BUFFER_BIT;

    if (this.options_.depth) {
      mask |= this.context.DEPTH_BUFFER_BIT;
    }
    if (this.options_.stencil) {
      mask |= this.context.STENCIL_BUFFER_BIT;
    }

    this.context.clear(mask);
  }

  /**
   * Sets blend state.
   * @param {boolean} enabled - Whether blending is enabled.
   * @param {Object} [func] - Blend function configuration.
   */
  setBlendState(enabled, func = this.blendFunc_) {
    this.blendEnabled_ = enabled;

    if (enabled) {
      this.context.enable(this.context.BLEND);
      this.context.blendFunc(
        func.src || this.context.SRC_ALPHA,
        func.dst || this.context.ONE_MINUS_SRC_ALPHA
      );
    } else {
      this.context.disable(this.context.BLEND);
    }
  }

  /**
   * Sets depth test state.
   * @param {boolean} enabled - Whether depth testing is enabled.
   * @param {number} [func=GL.LESS] - Depth function.
   */
  setDepthTest(enabled, func = this.context.LESS) {
    this.depthTestEnabled_ = enabled;
    this.depthFunc_ = func;

    if (enabled) {
      this.context.enable(this.context.DEPTH_TEST);
      this.context.depthFunc(func);
    } else {
      this.context.disable(this.context.DEPTH_TEST);
    }
  }

  /**
   * Sets face culling.
   * @param {boolean} enabled - Whether face culling is enabled.
   * @param {number} [mode=GL.BACK] - Which face to cull.
   */
  setCullFace(enabled, mode = this.context.BACK) {
    this.cullFaceEnabled_ = enabled;
    this.cullFaceMode_ = mode;

    if (enabled) {
      this.context.enable(this.context.CULL_FACE);
      this.context.cullFace(mode);
    } else {
      this.context.disable(this.context.CULL_FACE);
    }
  }

  /**
   * Sets the current program for rendering.
   * @param {string} name - Program name.
   */
  useProgram(name) {
    const programMetadata = this.programs_.get(name);
    if (!programMetadata) {
      throw new Error(`Program "${name}" not found.`);
    }

    if (this.currentProgram_ !== name) {
      this.context.useProgram(programMetadata.program);
      this.currentProgram_ = name;
      this.emit('programSet', { name });
    }
  }

  /**
   * Sets the current framebuffer.
   * @param {?string} name - Framebuffer name or null for default.
   */
  setFramebuffer(name) {
    if (name && !this.framebuffers_.has(name)) {
      throw new Error(`Framebuffer "${name}" not found.`);
    }

    const framebuffer = name ? this.framebuffers_.get(name) : null;
    this.context.bindFramebuffer(this.context.FRAMEBUFFER, framebuffer);
    this.currentFramebuffer_ = name;
  }

  /**
   * Sets the active texture unit.
   * @param {number} unit - Texture unit index.
   */
  setActiveTexture(unit) {
    this.context.activeTexture(this.context.TEXTURE0 + unit);
  }

  /**
   * Binds a texture.
   * @param {string} name - Texture name.
   * @param {number} [target=GL.TEXTURE_2D] - Texture target.
   */
  bindTexture(name, target = this.context.TEXTURE_2D) {
    const textureMetadata = this.textures_.get(name);
    if (!textureMetadata) {
      throw new Error(`Texture "${name}" not found.`);
    }
    this.context.bindTexture(target, textureMetadata.texture);
  }

  /**
   * Draws primitives.
   * @param {Object} drawCall - Draw call configuration.
   */
  draw(drawCall) {
    if (!this.isInitialized) {
      console.warn('Renderer not initialized. Call initialize() first.');
      return;
    }

    if (!this.currentProgram_) {
      throw new Error('No program set. Call useProgram() first.');
    }

    const {
      mode = this.context.TRIANGLES,
      count,
      indices,
      offset = 0,
      type = this.context.UNSIGNED_SHORT,
    } = drawCall;

    // Set up vertex attributes
    if (drawCall.attributes) {
      Object.entries(drawCall.attributes).forEach(
        ([bufferName, attributes]) => {
          const bufferMetadata = this.buffers_.get(bufferName);
          if (bufferMetadata) {
            this.context.bindBuffer(
              this.context.ARRAY_BUFFER,
              bufferMetadata.buffer
            );

            attributes.forEach(attr => {
              this.context.enableVertexAttribArray(attr.location);
              this.context.vertexAttribPointer(
                attr.location,
                attr.size,
                attr.type || this.context.FLOAT,
                attr.normalized || false,
                attr.stride || 0,
                attr.offset || 0
              );
            });
          }
        }
      );
    }

    // Draw call
    if (indices) {
      const bufferMetadata = this.buffers_.get(indices);
      if (bufferMetadata) {
        this.context.bindBuffer(
          this.context.ELEMENT_ARRAY_BUFFER,
          bufferMetadata.buffer
        );
        this.context.drawElements(mode, count, type, offset);
      }
    } else if (count) {
      this.context.drawArrays(mode, offset, count);
    }

    this.emit('draw', { drawCall });
  }

  /**
   * Sets the size of the rendering viewport.
   * @override
   * @param {number} width - Width in pixels.
   * @param {number} height - Height in pixels.
   */
  setSize(width, height) {
    const actualWidth = Math.max(1, width);
    const actualHeight = Math.max(1, height);

    this.canvas.width = actualWidth;
    this.canvas.height = actualHeight;

    this.context.viewport(0, 0, actualWidth, actualHeight);

    this.emit('resize', { width: actualWidth, height: actualHeight });
  }

  /**
   * Renders a frame.
   * @override
   */
  render() {
    this.frameCount_++;
    this.emit('frameRendered', { frameCount: this.frameCount_ });
  }

  /**
   * Destroys all resources and cleans up the renderer.
   * @override
   */
  destroy() {
    // Clean up programs
    this.programs_.forEach(metadata => {
      this.context.deleteProgram(metadata.program);
      this.context.deleteShader(metadata.vertexShader);
      this.context.deleteShader(metadata.fragmentShader);
    });
    this.programs_.clear();

    // Clean up buffers
    this.buffers_.forEach(metadata => {
      this.context.deleteBuffer(metadata.buffer);
    });
    this.buffers_.clear();

    // Clean up textures
    this.textures_.forEach(metadata => {
      this.context.deleteTexture(metadata.texture);
    });
    this.textures_.clear();

    // Clean up framebuffers
    this.framebuffers_.forEach(framebuffer => {
      this.context.deleteFramebuffer(framebuffer);
    });
    this.framebuffers_.clear();

    this.context = null;
    this.isInitialized = false;

    this.emit('destroyed');
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
      programs: this.programs_.size,
      framebuffers: this.framebuffers_.size,
    };
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
   * Checks if WebGL is supported in the current environment.
   * @override
   * @returns {boolean} True if WebGL is supported, otherwise false.
   */
  static isSupported() {
    const canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    return gl !== null && gl instanceof WebGLRenderingContext;
  }

  // Helper methods
  createVertexBuffer(name, vertices, usage = this.context.STATIC_DRAW) {
    return this.createBuffer({
      name,
      data: vertices,
      usage,
      type: this.context.ARRAY_BUFFER,
    });
  }

  createIndexBuffer(name, indices, usage = this.context.STATIC_DRAW) {
    return this.createBuffer({
      name,
      data: indices,
      usage,
      type: this.context.ELEMENT_ARRAY_BUFFER,
    });
  }

  getBuffer(name) {
    const metadata = this.buffers_.get(name);
    return metadata ? metadata.buffer : null;
  }

  getProgram(name) {
    const metadata = this.programs_.get(name);
    return metadata ? metadata.program : null;
  }

  getTexture(name) {
    const metadata = this.textures_.get(name);
    return metadata ? metadata.texture : null;
  }

  getFramebuffer(name) {
    return this.framebuffers_.get(name) || null;
  }
}

/** Returns HTML template for the error message
 * @param {Error} error - Caught error
 * @returns {string}
 */
export function HTMLWGLErrorMessage(error) {
  return `
        <div style="color: white; font-family: Arial; text-align: center; padding: 50px;">
            <h2>WebGL Error</h2>
            <p>Rendering code may be broken or unsupported.</p>
            <p style="color: #ff6b6b; margin-top: 20px;">Error: ${error.message}</p>
            <div style="margin-top: 30px;">
                <a href="https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext"
                   target="_blank"
                   style="color: #4ecdc4; text-decoration: none;">
                   Learn more about WebGL
                </a>
            </div>
        </div>
    `;
}

/** Returns HTML template for unsupported message
 * @param {string} msg - Additional message
 * @returns {string}
 */
export function HTMLWGLUnsupportedMessage(msg = '') {
  return `
        <div style="color: white; font-family: Arial; text-align: center; padding: 50px;">
            <h2>WebGL Not Supported</h2>
            <p>Your browser does not support WebGL or it's disabled.</p>
            <p>Please use a modern browser with WebGL support.</p>
            <p style="color: #6d6d6dff; margin-top: 20px;">Developer's note: ${msg}</p>
            <div style="margin-top: 30px;">
                <a href="https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext"
                   target="_blank"
                   style="color: #4ecdc4; text-decoration: none;">
                   Learn more about WebGL
                </a>
            </div>
        </div>
    `;
}
