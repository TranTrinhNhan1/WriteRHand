/** Browser/Node evaluator for the exported AdaBoost-SAMME ensemble (`models/adaboost.json`). */

export interface AdaBoostTree {
  f: number[]
  t: number[]
  l: number[]
  r: number[]
  c: number[]
}

export interface AdaBoostSpec {
  type: string
  classes: number[]
  alphas: number[]
  trees: AdaBoostTree[]
  preprocess: { mean: number; std: number; eps: number }
}

export function parseAdaBoostSpec(raw: unknown): AdaBoostSpec {
  const spec = raw as Partial<AdaBoostSpec> | null
  if (!spec || spec.type !== 'adaboost-samme' || !Array.isArray(spec.trees) || !Array.isArray(spec.alphas)) {
    throw new Error('Unexpected AdaBoost model file format.')
  }
  if (!spec.preprocess || !Array.isArray(spec.classes) || spec.trees.length !== spec.alphas.length) {
    throw new Error('AdaBoost model file is incomplete.')
  }
  const k = spec.classes.length
  for (const tree of spec.trees) {
    const n = tree.f.length
    if (![tree.t, tree.l, tree.r, tree.c].every((a) => a.length === n)) {
      throw new Error('AdaBoost tree arrays have inconsistent lengths.')
    }
    for (let i = 0; i < n; i++) {
      if (tree.f[i] < 0) {
        if (tree.c[i] < 0 || tree.c[i] >= k) throw new Error('AdaBoost leaf has an invalid class index.')
      } else if (tree.l[i] <= i || tree.r[i] <= i || tree.l[i] >= n || tree.r[i] >= n) {
        throw new Error('AdaBoost tree has an invalid child index.')
      }
    }
  }
  return spec as AdaBoostSpec
}

/**
 * Vote shares for one image. `pixels01` holds 784 raw pixels scaled to [0, 1];
 * the standardisation (p - mean) / (std + eps) is applied here in float64, as in Python.
 */
export function adaboostPredict(spec: AdaBoostSpec, pixels01: ArrayLike<number>): Float64Array {
  const { mean, std, eps } = spec.preprocess
  const denom = std + eps
  const x = new Float64Array(pixels01.length)
  for (let i = 0; i < x.length; i++) x[i] = (pixels01[i] - mean) / denom

  const scores = new Float64Array(spec.classes.length)
  let total = 0
  for (let t = 0; t < spec.trees.length; t++) {
    const tree = spec.trees[t]
    let node = 0
    while (tree.f[node] >= 0) {
      node = x[tree.f[node]] < tree.t[node] ? tree.l[node] : tree.r[node]
    }
    scores[tree.c[node]] += spec.alphas[t]
    total += spec.alphas[t]
  }
  if (total > 0) for (let k = 0; k < scores.length; k++) scores[k] /= total
  return scores
}

export function argmax(v: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i
  return best
}
