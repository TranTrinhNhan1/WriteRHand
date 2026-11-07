/**
 * MNIST-style preprocessing on a plain grayscale buffer (no DOM, runs in Node).
 *
 * Recipe (as used to build MNIST): take the bounding box of the ink, scale the longer side to
 * 20 px keeping the aspect ratio (area-averaging, so strokes stay anti-aliased), paste the result
 * into a 28x28 field and translate it so that the centre of mass sits at (14, 14).
 * Convention: white ink (1.0) on a black background (0.0), pixel i has coordinate i.
 */

export const MNIST_SIZE = 28
export const GLYPH_SIZE = 20
export const COM_TARGET = 14
/** A pixel counts as ink for the bounding box when it is brighter than this. */
export const INK_EPS = 0.04
/** Below this share of ink (mean value over the whole image) the input counts as empty. */
export const MIN_INK_FRACTION = 0.0015

/** Grayscale image with values in [0, 1], row-major. */
export interface GrayImage {
  data: Float32Array
  width: number
  height: number
}

export interface BBox {
  x0: number
  y0: number
  /** Exclusive. */
  x1: number
  /** Exclusive. */
  y1: number
}

export interface Preprocessed {
  /** 784 values in [0, 1], row-major, white digit on black. */
  pixels: Float32Array
  bbox: BBox
  /** Zoom applied to the cropped glyph (20 / longer side). */
  scale: number
  /** Centre of mass of the final 28x28 image. */
  com: { x: number; y: number }
  /** Share of the source image covered by ink (0..1). */
  inkFraction: number
}

export function luminance(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255
}

/** Converts RGBA bytes (e.g. canvas ImageData) to a grayscale image in [0, 1]. */
export function grayFromRGBA(rgba: ArrayLike<number>, width: number, height: number): GrayImage {
  const data = new Float32Array(width * height)
  for (let i = 0; i < data.length; i++) {
    data[i] = luminance(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2])
  }
  return { data, width, height }
}

export function meanValue(img: GrayImage): number {
  let s = 0
  for (let i = 0; i < img.data.length; i++) s += img.data[i]
  return img.data.length ? s / img.data.length : 0
}

/** Light backgrounds (dark ink on paper) are detected by a high mean luminance. */
export function shouldInvert(img: GrayImage): boolean {
  return meanValue(img) > 0.5
}

export function invert(img: GrayImage): GrayImage {
  const data = new Float32Array(img.data.length)
  for (let i = 0; i < data.length; i++) data[i] = 1 - img.data[i]
  return { data, width: img.width, height: img.height }
}

export interface UploadResult {
  image: GrayImage
  inverted: boolean
}

/**
 * Brings a photo or scan to "white ink on black": inverts light backgrounds, removes the
 * background level (median), stretches the contrast and gates out faint noise. A flat image
 * with no contrast comes back all zero (reported as empty by `preprocessMnist`).
 */
export function normalizeUpload(gray: GrayImage): UploadResult {
  const inverted = shouldInvert(gray)
  const src = inverted ? invert(gray) : gray
  const n = src.data.length
  const bins = new Uint32Array(256)
  let max = 0
  for (let i = 0; i < n; i++) {
    const v = src.data[i]
    bins[Math.min(255, Math.max(0, Math.round(v * 255)))]++
    if (v > max) max = v
  }
  let acc = 0
  let bg = 0
  for (let b = 0; b < 256; b++) {
    acc += bins[b]
    if (acc >= n / 2) {
      bg = b / 255
      break
    }
  }
  const range = max - bg
  const out = new Float32Array(n)
  if (range >= 0.15) {
    for (let i = 0; i < n; i++) {
      const v = (src.data[i] - bg) / range
      out[i] = v < 0.1 ? 0 : Math.min(1, v)
    }
  }
  return { image: { data: out, width: gray.width, height: gray.height }, inverted }
}

export function inkBBox(img: GrayImage, eps = INK_EPS): BBox | null {
  const { data, width, height } = img
  let x0 = width
  let y0 = height
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < height; y++) {
    const row = y * width
    for (let x = 0; x < width; x++) {
      if (data[row + x] > eps) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1: x1 + 1, y1: y1 + 1 }
}

interface AxisWeights {
  idx: Int32Array
  w: Float32Array
  /** Taps per output sample (taps beyond a sample's own count have weight 0). */
  taps: number
}

/** Area-averaging (box) weights when shrinking, linear interpolation when enlarging. */
function axisWeights(srcLen: number, dstLen: number): AxisWeights {
  if (dstLen <= srcLen) {
    const r = srcLen / dstLen
    const taps = Math.ceil(r) + 1
    const idx = new Int32Array(dstLen * taps)
    const w = new Float32Array(dstLen * taps)
    for (let i = 0; i < dstLen; i++) {
      const a = i * r
      const b = (i + 1) * r
      const first = Math.floor(a)
      for (let k = 0; k < taps; k++) {
        const s = first + k
        if (s >= srcLen) {
          idx[i * taps + k] = srcLen - 1
          continue
        }
        const overlap = Math.min(b, s + 1) - Math.max(a, s)
        idx[i * taps + k] = s
        w[i * taps + k] = overlap > 0 ? overlap / r : 0
      }
    }
    return { idx, w, taps }
  }
  const idx = new Int32Array(dstLen * 2)
  const w = new Float32Array(dstLen * 2)
  const r = srcLen / dstLen
  for (let i = 0; i < dstLen; i++) {
    const pos = (i + 0.5) * r - 0.5
    const i0 = Math.floor(pos)
    const f = pos - i0
    idx[2 * i] = Math.min(srcLen - 1, Math.max(0, i0))
    idx[2 * i + 1] = Math.min(srcLen - 1, Math.max(0, i0 + 1))
    w[2 * i] = 1 - f
    w[2 * i + 1] = f
  }
  return { idx, w, taps: 2 }
}

/** Separable resize of a w x h window of `src` starting at (x0, y0). */
export function resizeWindow(
  src: GrayImage,
  win: BBox,
  dstW: number,
  dstH: number,
): GrayImage {
  const sw = win.x1 - win.x0
  const sh = win.y1 - win.y0
  const wx = axisWeights(sw, dstW)
  const wy = axisWeights(sh, dstH)
  // Horizontal pass: sh rows x dstW columns.
  const tmp = new Float32Array(sh * dstW)
  for (let y = 0; y < sh; y++) {
    const rowOff = (win.y0 + y) * src.width + win.x0
    for (let x = 0; x < dstW; x++) {
      let acc = 0
      for (let k = 0; k < wx.taps; k++) {
        acc += src.data[rowOff + wx.idx[x * wx.taps + k]] * wx.w[x * wx.taps + k]
      }
      tmp[y * dstW + x] = acc
    }
  }
  const out = new Float32Array(dstH * dstW)
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      let acc = 0
      for (let k = 0; k < wy.taps; k++) {
        acc += tmp[wy.idx[y * wy.taps + k] * dstW + x] * wy.w[y * wy.taps + k]
      }
      out[y * dstW + x] = acc
    }
  }
  return { data: out, width: dstW, height: dstH }
}

export function centerOfMass(data: ArrayLike<number>, width: number, height: number) {
  let m = 0
  let sx = 0
  let sy = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[y * width + x]
      m += v
      sx += v * x
      sy += v * y
    }
  }
  return m > 0 ? { x: sx / m, y: sy / m } : { x: (width - 1) / 2, y: (height - 1) / 2 }
}

/** Translates by a (possibly fractional) offset with bilinear sampling; outside is black. */
function translate(data: Float32Array, size: number, dx: number, dy: number): Float32Array {
  const out = new Float32Array(size * size)
  const at = (x: number, y: number) =>
    x < 0 || y < 0 || x >= size || y >= size ? 0 : data[y * size + x]
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = x - dx
      const sy = y - dy
      const x0 = Math.floor(sx)
      const y0 = Math.floor(sy)
      const fx = sx - x0
      const fy = sy - y0
      out[y * size + x] =
        at(x0, y0) * (1 - fx) * (1 - fy) +
        at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy +
        at(x0 + 1, y0 + 1) * fx * fy
    }
  }
  return out
}

/**
 * Runs the MNIST recipe on a white-on-black grayscale image.
 * Returns null when the image holds (almost) no ink.
 */
export function preprocessMnist(img: GrayImage): Preprocessed | null {
  const { data, width, height } = img
  if (width < 1 || height < 1 || data.length !== width * height) return null

  let total = 0
  for (let i = 0; i < data.length; i++) total += data[i]
  const inkFraction = total / data.length
  if (inkFraction < MIN_INK_FRACTION) return null

  const bbox = inkBBox(img)
  if (!bbox) return null

  const bw = bbox.x1 - bbox.x0
  const bh = bbox.y1 - bbox.y0
  const scale = GLYPH_SIZE / Math.max(bw, bh)
  const dw = Math.max(1, Math.min(GLYPH_SIZE, Math.round(bw * scale)))
  const dh = Math.max(1, Math.min(GLYPH_SIZE, Math.round(bh * scale)))
  const glyph = resizeWindow(img, bbox, dw, dh)

  const field = new Float32Array(MNIST_SIZE * MNIST_SIZE)
  const ox = Math.round((MNIST_SIZE - dw) / 2)
  const oy = Math.round((MNIST_SIZE - dh) / 2)
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) field[(oy + y) * MNIST_SIZE + ox + x] = glyph.data[y * dw + x]
  }

  const c0 = centerOfMass(field, MNIST_SIZE, MNIST_SIZE)
  let pixels = translate(field, MNIST_SIZE, COM_TARGET - c0.x, COM_TARGET - c0.y)

  let peak = 0
  for (let i = 0; i < pixels.length; i++) if (pixels[i] > peak) peak = pixels[i]
  if (peak <= 0) return null
  if (peak < 1) {
    pixels = pixels.map((v) => Math.min(1, v / peak))
  }
  return {
    pixels,
    bbox,
    scale,
    com: centerOfMass(pixels, MNIST_SIZE, MNIST_SIZE),
    inkFraction,
  }
}

/** Nearest-neighbour enlargement of an n x n image by an integer factor (used for the round-trip test). */
export function upsample(data: ArrayLike<number>, n: number, factor: number): GrayImage {
  const size = n * factor
  const out = new Float32Array(size * size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      out[y * size + x] = data[Math.floor(y / factor) * n + Math.floor(x / factor)]
    }
  }
  return { data: out, width: size, height: size }
}
