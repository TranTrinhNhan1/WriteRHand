/**
 * Recomputation of selected quantities from the learned parameters (float64 accumulation).
 * The explorer shows these numbers next to the values the deployed ONNX graph returned and the
 * absolute difference, so every displayed number can be checked by the reader.
 */

export interface Check {
  /** value rebuilt from the displayed components */
  rebuilt: number
  /** value returned by the deployed graph */
  runtime: number
  absDiff: number
}

export const check = (rebuilt: number, runtime: number): Check => ({ rebuilt, runtime, absDiff: Math.abs(rebuilt - runtime) })

export function softmax(z: ArrayLike<number>): Float64Array {
  let m = -Infinity
  for (let i = 0; i < z.length; i++) m = Math.max(m, z[i])
  const e = Float64Array.from(z as ArrayLike<number>, (v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function argmax(v: ArrayLike<number>): number {
  let b = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i
  return b
}

// --------------------------------------------------------------------------- dense layers
export interface DenseNeuron {
  /** x_i * W_ij for every input i */
  terms: Float64Array
  positive: number
  negative: number
  bias: number
  /** sum of terms + bias */
  z: number
}

/** Weighted sum of one output neuron j of a dense layer. `W` is [in, out] row-major. */
export function denseNeuron(input: ArrayLike<number>, W: Float32Array, b: Float32Array, nIn: number, nOut: number, j: number): DenseNeuron {
  const terms = new Float64Array(nIn)
  let pos = 0
  let neg = 0
  for (let i = 0; i < nIn; i++) {
    const t = input[i] * W[i * nOut + j]
    terms[i] = t
    if (t > 0) pos += t
    else neg += t
  }
  return { terms, positive: pos, negative: neg, bias: b[j], z: pos + neg + b[j] }
}

/** Full dense layer pre-activation z = a W + b. */
export function denseForward(input: ArrayLike<number>, W: Float32Array, b: Float32Array, nIn: number, nOut: number): Float64Array {
  const z = Float64Array.from(b)
  for (let i = 0; i < nIn; i++) {
    const v = input[i]
    if (v === 0) continue
    const row = i * nOut
    for (let j = 0; j < nOut; j++) z[j] += v * W[row + j]
  }
  return z
}

// --------------------------------------------------------------------------- convolution
export interface ConvPatch {
  /** patch values [Cin][kh][kw] flattened, kernel values, and their products */
  patch: Float64Array
  kernel: Float64Array
  products: Float64Array
  sum: number
  bias: number
  /** sum + bias = pre-activation at (c, r, q) */
  z: number
  /** input rows/cols covered (inclusive start, exclusive end) */
  y0: number
  x0: number
  kh: number
  kw: number
}

/**
 * One output value of a convolution (cross-correlation, stride s, no padding):
 * `z[c,r,q] = b[c] + sum_{ci,i,j} in[ci, r*s+i, q*s+j] * K[c,ci,i,j]`.
 * `input` is [Cin, H, W] row-major.
 */
export function convPatch(input: ArrayLike<number>, inShape: number[], K: Float32Array, kShape: number[], bias: Float32Array, c: number, r: number, q: number, stride = 1): ConvPatch {
  const [, H, W] = inShape
  const [, cin, kh, kw] = kShape
  const n = cin * kh * kw
  const patch = new Float64Array(n)
  const kernel = new Float64Array(n)
  const products = new Float64Array(n)
  let sum = 0
  let p = 0
  for (let ci = 0; ci < cin; ci++) {
    for (let i = 0; i < kh; i++) {
      for (let j = 0; j < kw; j++, p++) {
        patch[p] = input[ci * H * W + (r * stride + i) * W + (q * stride + j)]
        kernel[p] = K[((c * cin + ci) * kh + i) * kw + j]
        products[p] = patch[p] * kernel[p]
        sum += products[p]
      }
    }
  }
  return { patch, kernel, products, sum, bias: bias[c], z: sum + bias[c], y0: r * stride, x0: q * stride, kh, kw }
}

export interface PoolWindow {
  values: Float64Array
  /** index of the maximum inside the window (row-major), its (dy, dx), and the value */
  argmax: number
  dy: number
  dx: number
  max: number
  y0: number
  x0: number
  size: number
}

/** `p x p` max-pool window of channel `c` at output location (r, q). `act` is [C, H, W]. */
export function poolWindow(act: ArrayLike<number>, shape: number[], c: number, r: number, q: number, p: number, stride = p): PoolWindow {
  const [, H, W] = shape
  const values = new Float64Array(p * p)
  let best = 0
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      values[i * p + j] = act[c * H * W + (r * stride + i) * W + (q * stride + j)]
      if (values[i * p + j] > values[best]) best = i * p + j
    }
  }
  return { values, argmax: best, dy: Math.floor(best / p), dx: best % p, max: values[best], y0: r * stride, x0: q * stride, size: p }
}

export interface StageGeometry {
  kind: 'conv2d' | 'maxpool2d'
  size: number
  stride: number
}

/**
 * Receptive field in the network input of the output location (r, q) of the layer at the END of
 * `stack` (stack lists the conv/pool layers from the input up to and including that layer).
 * Returns inclusive-exclusive pixel bounds. Exact for padding 0.
 */
export function receptiveField(stack: StageGeometry[], r: number, q: number): { y0: number; y1: number; x0: number; x1: number } {
  let y0 = r
  let y1 = r
  let x0 = q
  let x1 = q
  for (let i = stack.length - 1; i >= 0; i--) {
    const { size, stride } = stack[i]
    y0 = y0 * stride
    x0 = x0 * stride
    y1 = y1 * stride + size - 1
    x1 = x1 * stride + size - 1
  }
  return { y0, y1: y1 + 1, x0, x1: x1 + 1 }
}

// --------------------------------------------------------------------------- Gaussian Naive Bayes
export interface NbTerms {
  /** per-pixel log N(x_j; mu_kj, var_kj) = -0.5 [log(2 pi var) + (x - mu)^2 / var] */
  perPixel: Float64Array
  /** the two pieces of that term, for the explanation */
  logNorm: Float64Array
  quad: Float64Array
  logPrior: number
  sum: number
  /** log prior + sum of per-pixel terms = log joint of class k */
  logJoint: number
}

export function nbClassTerms(x: ArrayLike<number>, mean: Float32Array, variance: Float32Array, logPrior: Float32Array, k: number, nFeatures = x.length): NbTerms {
  const perPixel = new Float64Array(nFeatures)
  const logNorm = new Float64Array(nFeatures)
  const quad = new Float64Array(nFeatures)
  let sum = 0
  const base = k * nFeatures
  for (let j = 0; j < nFeatures; j++) {
    const v = variance[base + j]
    const d = x[j] - mean[base + j]
    logNorm[j] = -0.5 * Math.log(2 * Math.PI * v)
    quad[j] = (-0.5 * d * d) / v
    perPixel[j] = logNorm[j] + quad[j]
    sum += perPixel[j]
  }
  return { perPixel, logNorm, quad, logPrior: logPrior[k], sum, logJoint: logPrior[k] + sum }
}

/** log-sum-exp, used to turn log joints into the evidence that normalises the posterior. */
export function logSumExp(v: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < v.length; i++) m = Math.max(m, v[i])
  let s = 0
  for (let i = 0; i < v.length; i++) s += Math.exp(v[i] - m)
  return m + Math.log(s)
}

// --------------------------------------------------------------------------- softmax regression
export interface SoftmaxTerms {
  /** x_i * W_ik */
  terms: Float64Array
  bias: number
  logit: number
}

export function softmaxClassTerms(x: ArrayLike<number>, W: Float32Array, b: Float32Array, nIn: number, nOut: number, k: number): SoftmaxTerms {
  const n = denseNeuron(x, W, b, nIn, nOut, k)
  return { terms: n.terms, bias: n.bias, logit: n.z }
}

/** Running sum of `terms` in a given pixel order: the curve behind the "evidence accumulates" animation. */
export function runningSum(terms: ArrayLike<number>, order: ArrayLike<number>, start = 0): Float64Array {
  const out = new Float64Array(order.length + 1)
  out[0] = start
  for (let i = 0; i < order.length; i++) out[i + 1] = out[i] + terms[order[i]]
  return out
}

/** Pixel indices sorted by decreasing |term|: the order in which the biggest contributions are applied first. */
export function orderByMagnitude(terms: ArrayLike<number>): Int32Array {
  const idx = Int32Array.from({ length: terms.length }, (_, i) => i)
  return idx.sort((a, b) => Math.abs(terms[b]) - Math.abs(terms[a]))
}
