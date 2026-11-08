/** What happens inside each classifier for one image: the data behind the "look inside" panel. */
import type { AdaBoostSpec } from './adaboost'
import type { ModelId } from './types'

export const N_PIXELS = 784

export interface NaiveBayesTrace {
  kind: 'naive_bayes'
  /** Standardised input, 784 values. */
  x: Float32Array
  /** Per class, per pixel Gaussian parameters (standardised units), row-major [class][pixel]. */
  mean: Float32Array
  variance: Float32Array
  logPrior: Float32Array
  /** log p(class) + sum of per-pixel log-likelihoods, before softmax. */
  logits: Float32Array
}

export interface SoftmaxTrace {
  kind: 'softmax'
  x: Float32Array
  /** [pixel][class] as stored by the model. */
  weights: Float32Array
  bias: Float32Array
  logits: Float32Array
}

export interface MlpTrace {
  kind: 'mlp'
  hidden: Float32Array[]
  logits: Float32Array
}

export interface PathStep {
  node: number
  feature: number
  threshold: number
  /** Raw pixel value in [0, 1] and the threshold in the same units. */
  value: number
  goLeft: boolean
  /** Number of nodes in the branch that was not taken. */
  otherSize: number
}

export interface AdaRound {
  alpha: number
  digit: number
  path: PathStep[]
}

export interface AdaBoostTrace {
  kind: 'adaboost'
  x: Float32Array
  classes: number[]
  rounds: AdaRound[]
}

export type Trace = NaiveBayesTrace | SoftmaxTrace | MlpTrace | AdaBoostTrace

/** ONNX outputs to request for each model (besides `probabilities`). */
export const TRACE_OUTPUTS: Partial<Record<ModelId, string[]>> = {
  naive_bayes: ['x', 'class_mean', 'class_var', 'log_prior', 'logits'],
  softmax: ['x', 'weights', 'bias', 'logits'],
  mlp: ['a0', 'a1', 'logits'],
}

function subtreeSize(tree: AdaBoostSpec['trees'][number], root: number): number {
  let n = 0
  const stack = [root]
  while (stack.length) {
    const i = stack.pop()!
    n++
    if (tree.f[i] >= 0) stack.push(tree.l[i], tree.r[i])
  }
  return n
}

/** Walks every tree of the ensemble for this image and records the path each one takes. */
export function adaboostTrace(spec: AdaBoostSpec, pixels01: ArrayLike<number>): AdaBoostTrace {
  const { mean, std, eps } = spec.preprocess
  const denom = std + eps
  const x = new Float32Array(N_PIXELS)
  for (let i = 0; i < N_PIXELS; i++) x[i] = (pixels01[i] - mean) / denom
  const rounds: AdaRound[] = spec.trees.map((tree, t) => {
    const path: PathStep[] = []
    let node = 0
    while (tree.f[node] >= 0) {
      const feature = tree.f[node]
      const goLeft = x[feature] < tree.t[node]
      const next = goLeft ? tree.l[node] : tree.r[node]
      const other = goLeft ? tree.r[node] : tree.l[node]
      path.push({ node, feature, threshold: tree.t[node] * denom + mean, value: pixels01[feature], goLeft, otherSize: subtreeSize(tree, other) })
      node = next
    }
    return { alpha: spec.alphas[t], digit: spec.classes[tree.c[node]], path }
  })
  return { kind: 'adaboost', x, classes: spec.classes, rounds }
}
