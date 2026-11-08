/**
 * Reader and evaluator for `adaboost.bin` (format: `adaboost_to_binary` in src/export/onnx_export.py).
 * The ensemble is exposed as typed-array views over the downloaded buffer: no JSON parse, no
 * per-node objects, so a 400-tree / 600 000-node model costs one allocation per tree array.
 */

const MAGIC = 'WRHADA01'

export interface Tree {
  /** split threshold in standardised units (float64) */
  t: Float64Array
  /** feature index, -1 for a leaf */
  f: Int16Array
  l: Int16Array
  r: Int16Array
  /** leaf class index */
  c: Uint8Array
}

export interface AdaBoostEnsemble {
  classes: number[]
  alphas: number[]
  preprocess: { mean: number; std: number; eps: number }
  nodeCounts: number[]
  depths: number[]
  trees: Tree[]
  totalNodes: number
  _sizes: (Int32Array | undefined)[]
}

const pad8 = (n: number) => n + ((8 - (n % 8)) % 8)

export function parseAdaBoostBinary(buf: ArrayBuffer): AdaBoostEnsemble {
  const head = new Uint8Array(buf, 0, 8)
  if (String.fromCharCode(...head) !== MAGIC) throw new Error('not a WriteRHand AdaBoost file')
  const hlen = new DataView(buf).getUint32(8, true)
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hlen)))
  let pos = 12 + hlen
  const trees: Tree[] = []
  for (const n of header.node_counts as number[]) {
    const t = new Float64Array(buf, pos, n)
    pos += pad8(n * 8)
    const f = new Int16Array(buf, pos, n)
    pos += pad8(n * 2)
    const l = new Int16Array(buf, pos, n)
    pos += pad8(n * 2)
    const r = new Int16Array(buf, pos, n)
    pos += pad8(n * 2)
    const c = new Uint8Array(buf, pos, n)
    pos += pad8(n)
    trees.push({ t, f, l, r, c })
  }
  if (pos !== buf.byteLength) throw new Error('AdaBoost file has trailing or missing bytes')
  const k = header.classes.length
  if (header.alphas.length !== trees.length) throw new Error('AdaBoost header disagrees with its trees')
  for (const tree of trees) {
    for (let i = 0; i < tree.f.length; i++) {
      if (tree.f[i] < 0) {
        if (tree.c[i] >= k) throw new Error('AdaBoost leaf has an invalid class index')
      } else if (tree.l[i] <= i || tree.r[i] <= i || tree.l[i] >= tree.f.length || tree.r[i] >= tree.f.length) {
        throw new Error('AdaBoost tree has an invalid child index')
      }
    }
  }
  return {
    classes: header.classes,
    alphas: header.alphas,
    preprocess: header.preprocess,
    nodeCounts: header.node_counts,
    depths: header.depths,
    trees,
    totalNodes: (header.node_counts as number[]).reduce((a, b) => a + b, 0),
    _sizes: new Array(trees.length),
  }
}

export function standardize(ens: AdaBoostEnsemble, pixels01: ArrayLike<number>): Float64Array {
  const { mean, std, eps } = ens.preprocess
  const denom = std + eps
  const x = new Float64Array(pixels01.length)
  for (let i = 0; i < x.length; i++) x[i] = (pixels01[i] - mean) / denom
  return x
}

/** Leaf reached by `x` (standardised) in a tree, with the route taken. */
function walk(tree: Tree, x: Float64Array): { leaf: number; route: number[] } {
  const route: number[] = []
  let node = 0
  while (tree.f[node] >= 0) {
    route.push(node)
    node = x[tree.f[node]] < tree.t[node] ? tree.l[node] : tree.r[node]
  }
  return { leaf: node, route }
}

/** Vote shares of the ensemble (weighted votes divided by their sum). NOT calibrated probabilities. */
export function adaboostPredict(ens: AdaBoostEnsemble, pixels01: ArrayLike<number>): Float64Array {
  const x = standardize(ens, pixels01)
  const scores = new Float64Array(ens.classes.length)
  let total = 0
  for (let t = 0; t < ens.trees.length; t++) {
    const tree = ens.trees[t]
    let node = 0
    while (tree.f[node] >= 0) node = x[tree.f[node]] < tree.t[node] ? tree.l[node] : tree.r[node]
    scores[tree.c[node]] += ens.alphas[t]
    total += ens.alphas[t]
  }
  if (total > 0) for (let k = 0; k < scores.length; k++) scores[k] /= total
  return scores
}

/** Number of nodes in the subtree below each node (index > parent, so one reverse pass suffices). */
export function subtreeSizes(ens: AdaBoostEnsemble, t: number): Int32Array {
  let s = ens._sizes[t]
  if (!s) {
    const tree = ens.trees[t]
    s = new Int32Array(tree.f.length)
    for (let i = tree.f.length - 1; i >= 0; i--) s[i] = tree.f[i] < 0 ? 1 : 1 + s[tree.l[i]] + s[tree.r[i]]
    ens._sizes[t] = s
  }
  return s
}

export interface PathStep {
  node: number
  feature: number
  /** threshold in standardised units, and in pixel units (0..1) */
  threshold: number
  thresholdPixel: number
  /** input value at that feature: standardised and raw pixel */
  xStd: number
  pixel: number
  goLeft: boolean
  /** child taken / child skipped, and the size of the skipped branch */
  next: number
  other: number
  otherSize: number
}

export interface Round {
  index: number
  alpha: number
  /** predicted class index and the leaf that voted */
  klass: number
  leaf: number
  path: PathStep[]
}

export interface AdaBoostRun {
  rounds: Round[]
  /** cumulative[t][k] = sum of alphas of rounds 0..t-1 that voted for k  (length T + 1, first row zeros) */
  cumulative: Float64Array[]
  /** leader class after t rounds (-1 before any vote) and the final scores */
  leader: Int16Array
  scores: Float64Array
  shares: Float64Array
  top: number
}

/** Walks every tree for this image and records routes, votes and the running vote totals. */
export function adaboostRun(ens: AdaBoostEnsemble, pixels01: ArrayLike<number>): AdaBoostRun {
  const x = standardize(ens, pixels01)
  const { mean, std, eps } = ens.preprocess
  const denom = std + eps
  const K = ens.classes.length
  const rounds: Round[] = []
  const cumulative: Float64Array[] = [new Float64Array(K)]
  const leader = new Int16Array(ens.trees.length + 1).fill(-1)
  for (let t = 0; t < ens.trees.length; t++) {
    const tree = ens.trees[t]
    const { leaf, route } = walk(tree, x)
    const sizes = subtreeSizes(ens, t)
    const path: PathStep[] = route.map((node) => {
      const feature = tree.f[node]
      const goLeft = x[feature] < tree.t[node]
      const next = goLeft ? tree.l[node] : tree.r[node]
      const other = goLeft ? tree.r[node] : tree.l[node]
      return {
        node,
        feature,
        threshold: tree.t[node],
        thresholdPixel: tree.t[node] * denom + mean,
        xStd: x[feature],
        pixel: pixels01[feature],
        goLeft,
        next,
        other,
        otherSize: sizes[other],
      }
    })
    const klass = tree.c[leaf]
    rounds.push({ index: t, alpha: ens.alphas[t], klass, leaf, path })
    const row = Float64Array.from(cumulative[t])
    row[klass] += ens.alphas[t]
    cumulative.push(row)
    let best = 0
    for (let k = 1; k < K; k++) if (row[k] > row[best]) best = k
    leader[t + 1] = best
  }
  const scores = cumulative[cumulative.length - 1]
  const total = scores.reduce((a, b) => a + b, 0)
  const shares = Float64Array.from(scores, (v) => (total > 0 ? v / total : 0))
  let top = 0
  for (let k = 1; k < K; k++) if (shares[k] > shares[top]) top = k
  return { rounds, cumulative, leader, scores, shares, top }
}

export interface TreeNodeInfo {
  node: number
  depth: number
  parent: number
  isLeaf: boolean
  feature: number
  thresholdPixel: number
  klass: number
  size: number
}

/** Flat description of one tree (depth, parent, subtree size) for drawing; computed on demand. */
export function describeTree(ens: AdaBoostEnsemble, t: number): TreeNodeInfo[] {
  const tree = ens.trees[t]
  const { mean, std, eps } = ens.preprocess
  const sizes = subtreeSizes(ens, t)
  const out: TreeNodeInfo[] = new Array(tree.f.length)
  out[0] = { node: 0, depth: 0, parent: -1, isLeaf: tree.f[0] < 0, feature: tree.f[0], thresholdPixel: tree.t[0] * (std + eps) + mean, klass: tree.c[0], size: sizes[0] }
  for (let i = 0; i < tree.f.length; i++) {
    if (tree.f[i] < 0) continue
    for (const ch of [tree.l[i], tree.r[i]]) {
      out[ch] = {
        node: ch,
        depth: out[i].depth + 1,
        parent: i,
        isLeaf: tree.f[ch] < 0,
        feature: tree.f[ch],
        thresholdPixel: tree.t[ch] * (std + eps) + mean,
        klass: tree.c[ch],
        size: sizes[ch],
      }
    }
  }
  return out
}
