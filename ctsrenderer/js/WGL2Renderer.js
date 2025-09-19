/**
 * @fileoverview WebGL2 API based renderer implementation.
 * @copyright 2025–present Catsgold
 * @license GPL-3.0-or-later
 * You must credit Catsgold (me) if you use or modify this code.
 */

import { WGLRenderer } from './WGLRenderer.js';

/**
 * Renderer using WebGL2 API.
 * @extends WGLRenderer
 */
export class WGL2Renderer extends WGLRenderer {
  /**
   * Creates an instance of WGL2Renderer.
   * @param {HTMLCanvasElement} canvas - Canvas element for rendering.
   * @param {Object} [options] - Renderer options.
   */
  constructor(canvas, options = {}) {
    super(canvas, options);

    /**
     * Vertex array objects.
     * @type {Map<string, WebGLVertexArrayObject>}
     * @private
     */
    this.VAOs_ = new Map();

    /**
     * Current vertex array object.
     * @type {?string}
     * @private
     */
    this.currentVAO_ = null;
  }

  /**
   * Initializes WebGL2 renderer.
   * @override
   * @returns {Promise<boolean>} Resolves to true if successful.
   */
  async initialize() {
    if (!WGL2Renderer.isSupported()) {
      throw new Error('WebGL2 is not supported in this browser.');
    }

    try {
      this.context = this.canvas.getContext('webgl2', this.options_);
      if (!this.context) {
        throw new Error('Failed to get WebGL2 context.');
      }

      this.GL = this.context;

      // Set initial state (blend, depth, cull, clear color)
      this.setClearColor(...this.clearColor_);
      this.setDepthTest(this.depthTestEnabled_);
      this.setBlendState(this.blendEnabled_);
      this.setCullFace(this.cullFaceEnabled_, this.cullFaceMode_);

      this.isInitialized = true;
      this.emit('initialized', { gl: this.context, options: this.options_ });

      return true;
    } catch (error) {
      console.error('Failed to initialize WGL2Renderer:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Creates a vertex array object (VAO).
   * @param {string} name - Name of VAO.
   * @returns {WebGLVertexArrayObject} Created VAO.
   */
  createVAO(name) {
    if (!this.isInitialized) throw new Error('Renderer not initialized.');
    const vao = this.context.createVertexArray();
    this.VAOs_.set(name, vao);
    return vao;
  }

  /**
   * Binds a VAO.
   * @param {?string} name - VAO name, or null to unbind.
   */
  bindVAO(name) {
    if (name && !this.VAOs_.has(name)) {
      throw new Error(`VAO "${name}" not found.`);
    }
    this.context.bindVertexArray(name ? this.VAOs_.get(name) : null);
    this.currentVAO_ = name || null;
  }

  /**
   * Overrides draw to automatically use VAO if set.
   * @param {Object} drawCall - Draw call configuration.
   */
  draw(drawCall) {
    if (this.currentVAO_) {
      this.context.bindVertexArray(this.VAOs_.get(this.currentVAO_));
    }
    super.draw(drawCall);
    if (this.currentVAO_) {
      this.context.bindVertexArray(null);
    }
  }

  /**
   * Checks if WebGL2 is supported in the current environment.
   * @returns {boolean}
   */
  static isSupported() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    return gl !== null && gl instanceof WebGL2RenderingContext;
  }
}
