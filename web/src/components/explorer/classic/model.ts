/**
 * Pure helpers behind the three "classic" explorer views (Gaussian Naive Bayes, softmax regression,
 * AdaBoost). Nothing here invents a number: every function rearranges values that come from the
 * deployed model's parameters, its runtime outputs, or the parsed AdaBoost ensemble.
 */
import type { AdaBoostEnsemble, AdaBoostRun, Tree } from '../../../engine/adaboost'
import { orderByMagnitude } from '../../../engine/reconstruct'

// --------------------------------------------------------------------------- additive scores
/**
 * A class score that is a start value plus one term per input pixel:
 * Naive Bayes `log p(k) + sum_j log N(x_j; mu_kj, var_kj)`, softmax regression `b_k + sum_i x_i W_ik`.
 * `terms` is K x N row-major (row k = class k).
 */
export interface Additive {
  K: number
  N: number
  terms: Float64Array
  start: Float64Array
  /** start + sum of the class's terms, accumulated in float64 */
  total: Float64Array
}

export function makeAdditive(K: number, N: number, rowOf: (k: number) => { terms: ArrayLike<number>; start: number }): Additive {
  const terms = new Float64Array(K * N)
  const start = new Float64Array(K)
  const total = new Float64Array(K)
  for (let k = 0; k < K; k++) {
    const row = rowOf(k)
    if (row.terms.length !== N) throw new Error(`class ${k}: ${row.terms.length} terms, expected ${N}`)
    start[k] = row.start
    let s = row.start
    for (let j = 0; j < N; j++) {
      terms[k * N + j] = row.terms[j]
      s += row.terms[j]
    }
    total[k] = s
  }
  return { K, N, terms, start, total }
}

export const classRow = (a: Additive, k: number): Float64Array => a.terms.subarray(k * a.N, (k + 1) * a.N)

export type OrderKind = 'magnitude' | 'raster'

/** Order in which pixels are applied: raster (row by row) or by decreasing |term| of one class. */
export function pixelOrder(kind: OrderKind, classTerms: ArrayLike<number>): Int32Array {
  if (kind === 'raster') return Int32Array.from({ length: classTerms.length }, (_, i) => i)
  return orderByMagnitude(classTerms)
}

/**
 * Running score of every class after each step: row s (0..order.length) holds start + the terms of
 * the first s pixels of `order`. Returned row-major, (order.length + 1) x K.
 */
export function runningScores(a: Additive, order: ArrayLike<number>): Float64Array {
  const { K, N } = a
  const steps = order.length
  const out = new Float64Array((steps + 1) * K)
  out.set(a.start, 0)
  for (let s = 1; s <= steps; s++) {
    const j = order[s - 1]
    const prev = (s - 1) * K
    const cur = s * K
    for (let k = 0; k < K; k++) out[cur + k] = out[prev + k] + a.terms[k * N + j]
  }
  return out
}

/** Index of the largest value in row `s` of a (steps+1) x K matrix; -1 when every value is zero (no votes yet). */
export function rowLeader(rows: Float64Array, K: number, s: number, emptyIsNone = false): number {
  const base = s * K
  let best = 0
  let any = false
  for (let k = 0; k < K; k++) {
    if (rows[base + k] !== 0) any = true
    if (rows[base + k] > rows[base + best]) best = k
  }
  return emptyIsNone && !any ? -1 : best
}

export interface LeaderChange {
  step: number
  klass: number
}

/** Steps where the leader differs from the previous step's (the first defined leader included). */
export function leaderChanges(rows: Float64Array, K: number, emptyIsNone = false): LeaderChange[] {
  const steps = rows.length / K - 1
  const out: LeaderChange[] = []
  let prev = -2
  for (let s = 0; s <= steps; s++) {
    const l = rowLeader(rows, K, s, emptyIsNone)
    if (l !== prev && l >= 0) out.push({ step: s, klass: l })
    prev = l
  }
  return out
}

export function rowOf(rows: Float64Array, K: number, s: number): Float64Array {
  return rows.subarray(s * K, (s + 1) * K)
}

export function extent(v: ArrayLike<number>): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < v.length; i++) {
    if (v[i] < lo) lo = v[i]
    if (v[i] > hi) hi = v[i]
  }
  return [lo, hi]
}

/** Value at quantile q of |v| (used only to choose where a colour scale saturates; the scale is stated in the UI). */
export function absQuantile(v: ArrayLike<number>, q: number): number {
  const a = Float64Array.from(v as ArrayLike<number>, Math.abs).sort()
  if (!a.length) return 1
  return a[Math.min(a.length - 1, Math.floor(q * (a.length - 1)))] || a[a.length - 1] || 1
}

// --------------------------------------------------------------------------- Gaussian curves
export function gaussianPdf(x: number, mu: number, variance: number): number {
  return Math.exp((-0.5 * (x - mu) * (x - mu)) / variance) / Math.sqrt(2 * Math.PI * variance)
}

/** x range that shows +-4 standard deviations of every curve and the input value. */
export function curveDomain(mus: ArrayLike<number>, vars: ArrayLike<number>, x: number): [number, number] {
  let lo = x
  let hi = x
  for (let i = 0; i < mus.length; i++) {
    const sd = Math.sqrt(vars[i])
    lo = Math.min(lo, mus[i] - 4 * sd)
    hi = Math.max(hi, mus[i] + 4 * sd)
  }
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
}

export function sampleCurve(mu: number, variance: number, lo: number, hi: number, n: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = gaussianPdf(lo + ((hi - lo) * i) / (n - 1), mu, variance)
  return out
}

// --------------------------------------------------------------------------- AdaBoost trees
export interface CompactNode {
  /** node index in the tree, or the index of the first collapsed node for a stub */
  node: number
  depth: number
  kind: 'split' | 'leaf' | 'stub'
  /** index of the parent inside the compact list (-1 for the root) */
  parent: number
  /** which child of its parent: 'l' when x[feature] < threshold */
  side: 'l' | 'r' | null
  /** nodes represented: 1 for a drawn node, the subtree size for a stub */
  count: number
  /** subtree size of the real node (drawn nodes) */
  size: number
  onRoute: boolean
  feature: number
  /** threshold in pixel units 0..1 (split nodes) */
  thresholdPixel: number
  klass: number
  /** horizontal slot (leaves and stubs are 0, 1, 2 ...; splits sit at the middle of their children) */
  x: number
}

export interface CompactTree {
  nodes: CompactNode[]
  /** real nodes drawn, nodes folded into stubs, and the tree's total (shown + collapsed === total) */
  shown: number
  collapsed: number
  total: number
  slots: number
  maxDepth: number
}

/**
 * Compact drawing of one tree: every node at depth <= `fullDepth` and every node on the input's route
 * is drawn; each other subtree hanging off a drawn split becomes ONE stub that states how many real
 * nodes it stands for. Reads the typed arrays directly (no per-node objects for hidden nodes).
 */
export function compactTree(tree: Tree, sizes: Int32Array, route: ArrayLike<number>, fullDepth: number, toPixel: (t: number) => number): CompactTree {
  const onRoute = new Set<number>()
  for (let i = 0; i < route.length; i++) onRoute.add(route[i])
  const nodes: CompactNode[] = []
  let slot = 0
  let shown = 0
  let collapsed = 0
  let maxDepth = 0

  const visit = (node: number, depth: number, parent: number, side: 'l' | 'r' | null): number => {
    const idx = nodes.length
    maxDepth = Math.max(maxDepth, depth)
    const isLeaf = tree.f[node] < 0
    const visible = depth <= fullDepth || onRoute.has(node)
    if (!visible) {
      collapsed += sizes[node]
      nodes.push({ node, depth, kind: 'stub', parent, side, count: sizes[node], size: sizes[node], onRoute: false, feature: -1, thresholdPixel: NaN, klass: -1, x: slot++ })
      return idx
    }
    shown += 1
    nodes.push({
      node,
      depth,
      kind: isLeaf ? 'leaf' : 'split',
      parent,
      side,
      count: 1,
      size: sizes[node],
      onRoute: onRoute.has(node),
      feature: isLeaf ? -1 : tree.f[node],
      thresholdPixel: isLeaf ? NaN : toPixel(tree.t[node]),
      klass: isLeaf ? tree.c[node] : -1,
      x: 0,
    })
    if (isLeaf) {
      nodes[idx].x = slot++
    } else {
      const a = visit(tree.l[node], depth + 1, idx, 'l')
      const b = visit(tree.r[node], depth + 1, idx, 'r')
      nodes[idx].x = (nodes[a].x + nodes[b].x) / 2
    }
    return idx
  }
  visit(0, 0, -1, null)
  return { nodes, shown, collapsed, total: tree.f.length, slots: slot, maxDepth }
}

/** Number of split nodes per pixel in one tree. */
export function treeSplitCounts(tree: Tree, nFeatures: number): Float32Array {
  const out = new Float32Array(nFeatures)
  for (let i = 0; i < tree.f.length; i++) if (tree.f[i] >= 0) out[tree.f[i]] += 1
  return out
}

/**
 * Prefix counts of split nodes per pixel over the ensemble: row t (0..T) = splits on each pixel in
 * trees 0..t-1. Row-major (T + 1) x nFeatures.
 */
export function splitUsagePrefix(ens: AdaBoostEnsemble, nFeatures: number): Int32Array {
  const T = ens.trees.length
  const out = new Int32Array((T + 1) * nFeatures)
  for (let t = 0; t < T; t++) {
    const prev = t * nFeatures
    const cur = (t + 1) * nFeatures
    out.copyWithin(cur, prev, prev + nFeatures)
    const f = ens.trees[t].f
    for (let i = 0; i < f.length; i++) if (f[i] >= 0) out[cur + f[i]] += 1
  }
  return out
}

/** How often each pixel was tested on THIS input's routes in rounds 1..t. */
export function routeUsage(run: AdaBoostRun, t: number, nFeatures: number): Float32Array {
  const out = new Float32Array(nFeatures)
  for (let r = 0; r < t && r < run.rounds.length; r++) for (const s of run.rounds[r].path) out[s.feature] += 1
  return out
}

/** Vote shares after every round: row t = cumulative[t] / sum(cumulative[t]) (row 0 is all zero). */
export function shareSeries(cumulative: Float64Array[], K: number): Float64Array {
  const out = new Float64Array(cumulative.length * K)
  cumulative.forEach((row, t) => {
    let s = 0
    for (let k = 0; k < K; k++) s += row[k]
    if (s > 0) for (let k = 0; k < K; k++) out[t * K + k] = row[k] / s
  })
  return out
}

// --------------------------------------------------------------------------- formatting
const MINUS = '−'

/** Fixed digits with a typographic minus sign. */
export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '–'
  const s = Math.abs(v).toFixed(digits)
  return v < 0 && Number(s) !== 0 ? `${MINUS}${s}` : s
}

export function fmtSigned(v: number, digits = 3): string {
  const s = fmt(v, digits)
  return v > 0 && Number(Math.abs(v).toFixed(digits)) !== 0 ? `+${s}` : s
}

/** Small absolute differences: 0, or scientific notation like 3.1e−6. */
export function fmtDiff(v: number): string {
  if (!Number.isFinite(v)) return '–'
  if (v === 0) return '0'
  if (v >= 1e-2) return v.toFixed(3)
  const [m, e] = v.toExponential(1).split('e')
  return `${m}e${e.replace('-', MINUS).replace('+', '')}`
}

export const rowCol = (j: number, width = 28) => ({ r: Math.floor(j / width), c: j % width })
