/**
 * @fileoverview Abstract base class for all CTS renderers.
 * @copyright 2025–present Catsgold
 * @license GPL-3.0-or-later
 * You must credit Catsgold (me) if you use or modify this code.
 */

/**
 * Abstract base class for all CTS renderers.
 * Defines a common interface that all concrete renderers must implement.
 */
export class Renderer {
  /**
   * Creates a Renderer instance.
   * @param {HTMLCanvasElement} canvas - Canvas element for rendering.
   * @throws {Error} When trying to create an instance of an abstract class.
   */
  constructor(canvas) {
    if (new.target === Renderer) {
      throw new Error(
        'Renderer is an abstract class and cannot be instantiated directly.'
      );
    }

    /**
     * Canvas element associated with the renderer.
     * @type {HTMLCanvasElement}
     * @protected
     */
    this.canvas = canvas;

    /**
     * Rendering context (WebGL, WebGL2, WebGPU, etc.).
     * @type {?}
     * @protected
     */
    this.context = null;

    /**
     * Flag indicating whether the renderer is initialized.
     * @type {boolean}
     * @protected
     */
    this.isInitialized = false;
  }

  /**
   * Initializes the renderer.
   * @abstract
   * @returns {Promise<boolean>} A Promise that resolves to true upon successful initialization.
   * @throws {Error} If initialization failed.
   */
  async initialize() {
    throw new Error('Method "initialize()" must be implemented.');
  }

  /**
   * Sets the size of the rendering viewport.
   * @abstract
   * @param {number} width - Width in pixels.
   * @param {number} height - Height in pixels.
   */
  setSize(width, height) {
    throw new Error('Method "setSize()" must be implemented.');
  }

  /**
   * Sets the current clearColor.
   * @abstract
   * @param {number} r - Red cleaning color component (0-1).
   * @param {number} g - Green cleaning color component (0-1).
   * @param {number} b - Blue cleaning color component (0-1).
   * @param {number} a - Alpha cleaning color component (0-1).
   */
  setClearColor(r, g, b, a) {
    throw new Error('Method "setClearColor()" must be implemented.');
  }

  /**
   * Renders a frame. Must be implemented in derived classes.
   * @abstract
   */
  render() {
    throw new Error('Method "render()" must be implemented.');
  }

  /**
   * Checks if the given renderer type is supported in the current environment.
   * @abstract
   * @returns {boolean} True if supported, false otherwise.
   */
  static isSupported() {
    throw new Error('Method "isSupported()" must be implemented.');
  }
}
