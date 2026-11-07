import { describe, expect, it } from 'vitest'
import {
  COM_TARGET,
  GLYPH_SIZE,
  MNIST_SIZE,
  centerOfMass,
  grayFromRGBA,
  inkBBox,
  invert,
  normalizeUpload,
  preprocessMnist,
  shouldInvert,
  type GrayImage,
} from '../src/lib/preprocess'

function blank(w: number, h = w): GrayImage {
  return { data: new Float32Array(w * h), width: w, height: h }
}

/** Fills a thick line segment (a crude stroke) with value 1. */
function stroke(img: GrayImage, x0: number, y0: number, x1: number, y1: number, r: number) {
  const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0))
  for (let s = 0; s <= steps; s++) {
    const cx = x0 + ((x1 - x0) * s) / Math.max(steps, 1)
    const cy = y0 + ((y1 - y0) * s) / Math.max(steps, 1)
    for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++) {
      for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
        if (x >= 0 && y >= 0 && x < img.width && y < img.height && (x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
          img.data[y * img.width + x] = 1
        }
      }
    }
  }
}

/** An off-centre "7"-like glyph in a 280 px canvas. */
function offCentreSeven(): GrayImage {
  const img = blank(280)
  stroke(img, 150, 40, 260, 40, 10)
  stroke(img, 260, 40, 190, 200, 10)
  return img
}

function bboxOf(pixels: Float32Array, eps = 0.1) {
  let x0 = 99
  let y0 = 99
  let x1 = -1
  let y1 = -1
  for (let y = 0; y < MNIST_SIZE; y++) {
    for (let x = 0; x < MNIST_SIZE; x++) {
      if (pixels[y * MNIST_SIZE + x] > eps) {
        x0 = Math.min(x0, x)
        x1 = Math.max(x1, x)
        y0 = Math.min(y0, y)
        y1 = Math.max(y1, y)
      }
    }
  }
  return { w: x1 - x0 + 1, h: y1 - y0 + 1 }
}

describe('preprocessMnist', () => {
  it('returns a 28x28 image in [0, 1]', () => {
    const out = preprocessMnist(offCentreSeven())
    expect(out).not.toBeNull()
    expect(out!.pixels).toHaveLength(MNIST_SIZE * MNIST_SIZE)
    expect(Math.min(...out!.pixels)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...out!.pixels)).toBeLessThanOrEqual(1)
    expect(Math.max(...out!.pixels)).toBeCloseTo(1, 5)
  })

  it('moves the centre of mass of an off-centre glyph to (14, 14)', () => {
    const out = preprocessMnist(offCentreSeven())!
    const com = centerOfMass(out.pixels, MNIST_SIZE, MNIST_SIZE)
    expect(Math.abs(com.x - COM_TARGET)).toBeLessThan(0.5)
    expect(Math.abs(com.y - COM_TARGET)).toBeLessThan(0.5)
    expect(out.com.x).toBeCloseTo(com.x, 6)
  })

  it('centres glyphs placed in each corner of the canvas', () => {
    for (const [ox, oy] of [[5, 5], [200, 10], [10, 200], [200, 200]]) {
      const img = blank(280)
      stroke(img, ox + 10, oy + 10, ox + 60, oy + 10, 8)
      stroke(img, ox + 35, oy + 10, ox + 35, oy + 70, 8)
      const out = preprocessMnist(img)!
      expect(Math.abs(out.com.x - COM_TARGET)).toBeLessThan(0.5)
      expect(Math.abs(out.com.y - COM_TARGET)).toBeLessThan(0.5)
    }
  })

  it('scales the longer side of the ink box to 20 px', () => {
    const out = preprocessMnist(offCentreSeven())!
    const { w, h } = bboxOf(out.pixels)
    expect(Math.max(w, h)).toBeGreaterThanOrEqual(GLYPH_SIZE - 1)
    expect(Math.max(w, h)).toBeLessThanOrEqual(GLYPH_SIZE + 1)
    expect(out.scale).toBeCloseTo(GLYPH_SIZE / Math.max(out.bbox.x1 - out.bbox.x0, out.bbox.y1 - out.bbox.y0), 10)
  })

  it('scales the same way regardless of the size of the input glyph', () => {
    for (const k of [0.3, 0.6, 1]) {
      const img = blank(280)
      stroke(img, 140 - 100 * k, 140, 140 + 100 * k, 140, 8 * k + 2)
      stroke(img, 140, 140 - 100 * k, 140, 140 + 100 * k, 8 * k + 2)
      const { w, h } = bboxOf(preprocessMnist(img)!.pixels)
      expect(Math.max(w, h)).toBeGreaterThanOrEqual(19)
      expect(Math.max(w, h)).toBeLessThanOrEqual(21)
    }
  })

  it('preserves the aspect ratio of a wide and of a tall glyph', () => {
    const wide = blank(280)
    stroke(wide, 20, 100, 220, 100, 8)
    stroke(wide, 20, 150, 220, 150, 8)
    const w = bboxOf(preprocessMnist(wide)!.pixels)
    const srcRatio = (220 - 20 + 16) / (150 - 100 + 16)
    expect(w.w / w.h).toBeGreaterThan(srcRatio * 0.75)
    expect(w.w / w.h).toBeLessThan(srcRatio * 1.3)

    const tall = blank(280)
    stroke(tall, 100, 20, 100, 230, 7)
    const t = bboxOf(preprocessMnist(tall)!.pixels)
    expect(t.h).toBeGreaterThanOrEqual(19)
    expect(t.w).toBeLessThanOrEqual(4)
  })

  it('does not alias: a thin 1 px line shrunk by 10x keeps a soft, non-zero profile', () => {
    const img = blank(200)
    for (let y = 10; y < 190; y++) img.data[y * 200 + 100] = 1
    const out = preprocessMnist(img)!
    const row = Array.from(out.pixels.slice(14 * MNIST_SIZE, 15 * MNIST_SIZE))
    expect(row.filter((v) => v > 0).length).toBeGreaterThanOrEqual(1)
    expect(Math.max(...row)).toBeCloseTo(1, 5)
  })

  it('reports empty input as null', () => {
    expect(preprocessMnist(blank(280))).toBeNull()
    const speck = blank(280)
    speck.data[1000] = 1
    expect(preprocessMnist(speck)).toBeNull()
    const faint = blank(280)
    faint.data.fill(0.01)
    expect(preprocessMnist(faint)).toBeNull()
  })

  it('is deterministic', () => {
    const a = preprocessMnist(offCentreSeven())!
    const b = preprocessMnist(offCentreSeven())!
    expect(Array.from(a.pixels)).toEqual(Array.from(b.pixels))
  })

  it('handles non-square sources', () => {
    const img = blank(400, 120)
    stroke(img, 40, 60, 360, 60, 10)
    const out = preprocessMnist(img)!
    const { w, h } = bboxOf(out.pixels)
    expect(w).toBeGreaterThanOrEqual(19)
    expect(h).toBeLessThanOrEqual(4)
  })
})

describe('inkBBox', () => {
  it('finds the tight box with an exclusive upper bound', () => {
    const img = blank(50)
    stroke(img, 10, 20, 30, 20, 0)
    expect(inkBBox(img)).toEqual({ x0: 10, y0: 20, x1: 31, y1: 21 })
    expect(inkBBox(blank(10))).toBeNull()
  })
})

describe('upload normalisation', () => {
  it('detects light backgrounds and inverts them', () => {
    const paper = blank(100)
    paper.data.fill(0.95)
    stroke(paper, 20, 50, 80, 50, 4)
    for (let i = 0; i < paper.data.length; i++) if (paper.data[i] === 1) paper.data[i] = 0.1
    expect(shouldInvert(paper)).toBe(true)
    const { image, inverted } = normalizeUpload(paper)
    expect(inverted).toBe(true)
    expect(preprocessMnist(image)).not.toBeNull()
  })

  it('leaves dark backgrounds alone', () => {
    const board = blank(100)
    stroke(board, 20, 50, 80, 50, 4)
    expect(shouldInvert(board)).toBe(false)
    expect(normalizeUpload(board).inverted).toBe(false)
  })

  it('gives the same result for a digit and its inverted copy', () => {
    const dark = offCentreSeven()
    const light = invert(dark)
    const a = preprocessMnist(normalizeUpload(dark).image)!
    const b = preprocessMnist(normalizeUpload(light).image)!
    expect(Array.from(b.pixels)).toEqual(Array.from(a.pixels))
  })

  it('removes background noise before taking the bounding box', () => {
    const noisy = offCentreSeven()
    let seed = 7
    for (let i = 0; i < noisy.data.length; i++) {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      if (noisy.data[i] === 0) noisy.data[i] = (seed / 4294967296) * 0.06
    }
    const clean = preprocessMnist(offCentreSeven())!
    const out = preprocessMnist(normalizeUpload(noisy).image)!
    expect(out.bbox).toEqual(clean.bbox)
  })

  it('treats a flat image as empty', () => {
    const flat = blank(64)
    flat.data.fill(0.5)
    expect(preprocessMnist(normalizeUpload(flat).image)).toBeNull()
  })

  it('converts RGBA to luminance', () => {
    const g = grayFromRGBA([255, 255, 255, 255, 0, 0, 0, 255], 2, 1)
    expect(g.data[0]).toBeCloseTo(1, 5)
    expect(g.data[1]).toBe(0)
  })
})
