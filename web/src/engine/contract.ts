/**
 * Types and validation for the bundle manifest (`writerhand.bundle/2`) and its per-model trace
 * (`writerhand.trace/1`). The specification is docs/TRACE_CONTRACT.md. Validation is strict on
 * purpose: a manifest that does not describe a consistent model is rejected before anything is drawn.
 */

export const BUNDLE_SCHEMA = 'writerhand.bundle/2'
export const TRACE_SCHEMA = 'writerhand.trace/1'

export type ModelKind = 'naive_bayes' | 'softmax' | 'mlp' | 'cnn' | 'adaboost'
export const MODEL_ORDER: ModelKind[] = ['naive_bayes', 'softmax', 'mlp', 'cnn', 'adaboost']
export const isModelKind = (s: string): s is ModelKind => (MODEL_ORDER as string[]).includes(s)

export type LayerOp =
  | 'input'
  | 'flatten'
  | 'reshape'
  | 'standardize'
  | 'dense'
  | 'relu'
  | 'softmax'
  | 'conv2d'
  | 'maxpool2d'
  | 'gaussian_loglik'
  | 'tree_ensemble'
  | 'weighted_vote'
  | 'normalize'

export interface TraceLayer {
  id: string
  op: LayerOp
  inputs: string[]
  out_shape: number[]
  /** Name of the ONNX output that carries this layer's value for the current input, if any. */
  runtime?: string | null
  // op specific
  in_features?: number
  out_features?: number
  kernel_shape?: number[]
  stride?: number
  padding?: number
  window?: number
  mean?: number
  std?: number
  eps?: number
  n_classes?: number
  n_features?: number
  params?: Record<string, string>
  n_estimators?: number
  node_counts?: number[]
  depths?: number[]
}

export type RuntimeRole = 'standardized_input' | 'pre_activation' | 'activation' | 'pooled' | 'logits' | 'log_joint'

export interface RuntimeOutput {
  name: string
  layer: string
  shape: number[]
  role: RuntimeRole
}

export interface ParamTensor {
  name: string
  shape: number[]
  dtype: 'float32'
  offset: number
  nbytes: number
  layer: string
  role: string
}

export interface TraceParams {
  file: string
  sha256: string
  nbytes: number
  tensors: ParamTensor[]
}

export interface Trace {
  schema: string
  kind: ModelKind
  preprocessing: { op: string; [k: string]: unknown }[]
  input: { name: string; shape: (string | number)[]; range: [number, number] }
  layers: TraceLayer[]
  runtime_outputs: RuntimeOutput[]
  output: { name: string; kind: 'softmax_probability' | 'vote_share'; calibrated: boolean; classes: string[] }
  params: TraceParams | null
  n_parameters: number | null
}

export interface Validation {
  max_abs_prob_diff: number
  argmax_agreement: number
  n_checked: number
  ok: boolean
}

export interface ReplayRef {
  file: string
  sha256: string
  size_bytes: number
}

export interface ModelEntry {
  title: string
  class: string
  format: 'onnx' | 'binary'
  file: string
  sha256: string
  size_bytes: number
  params: Record<string, unknown>
  validation: Validation
  input: string
  output: string
  trace: Trace
  /** recorded training run (checkpoints, metrics) or null when the model has no iterative training */
  replay?: ReplayRef | null
}

export interface Preprocessing {
  version: string
  image: string
  scale: string
  mean: number
  std: number
  eps: number
  formula: string
}

export interface DatasetMeta {
  n_train: number
  n_val: number
  n_test: number
  seed: number
  split_fingerprint: string
  preprocess_version?: string
  dataset?: string
}

export interface DatasetEntry {
  title: string
  short: string
  kind: string
  source: string
  classes: string[]
  preprocessing: Preprocessing
  dataset: DatasetMeta
  test_vectors: string
  results: string
  models: Partial<Record<ModelKind, ModelEntry>>
}

export interface BundleManifest {
  schema: string
  name: string
  version: string
  created: string
  bundle_id: string
  default_dataset: string
  files_sha256: Record<string, string>
  datasets: Record<string, DatasetEntry>
}

export class ContractError extends Error {
  constructor(
    message: string,
    readonly where: string,
  ) {
    super(`${where}: ${message}`)
    this.name = 'ContractError'
  }
}

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)
const same = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v === b[i])

function fail(where: string, msg: string): never {
  throw new ContractError(msg, where)
}

/** Checks one model's trace: DAG order, shapes, parameter index, runtime outputs. */
export function validateTrace(trace: Trace, where: string): void {
  if (trace.schema !== TRACE_SCHEMA) fail(where, `unsupported trace schema ${String(trace.schema)}`)
  if (!isModelKind(trace.kind)) fail(where, `unknown model kind ${String(trace.kind)}`)
  const byId = new Map<string, TraceLayer>()
  for (const layer of trace.layers) {
    if (byId.has(layer.id)) fail(where, `duplicate layer id ${layer.id}`)
    for (const i of layer.inputs) if (!byId.has(i)) fail(where, `layer ${layer.id} reads ${i} before it is defined`)
    byId.set(layer.id, layer)
  }
  const classes = trace.output.classes.length
  const last = trace.layers[trace.layers.length - 1]
  if (!last || !same(last.out_shape, [classes])) fail(where, 'last layer must output one value per class')

  for (const layer of trace.layers) {
    const prev = layer.inputs.length ? byId.get(layer.inputs[0])! : null
    switch (layer.op) {
      case 'dense':
        if (!prev || prod(prev.out_shape) !== layer.in_features) fail(where, `${layer.id}: in_features != size of ${prev?.id}`)
        if (!same(layer.out_shape, [layer.out_features!])) fail(where, `${layer.id}: out_shape != out_features`)
        break
      case 'relu':
        if (!prev || !same(prev.out_shape, layer.out_shape)) fail(where, `${layer.id}: relu must keep the shape`)
        break
      case 'conv2d': {
        const k = layer.kernel_shape
        if (!prev || !k || k.length !== 4 || prev.out_shape.length !== 3) fail(where, `${layer.id}: bad convolution input`)
        const [cout, cin, kh, kw] = k
        const s = layer.stride ?? 1
        const pad = layer.padding ?? 0
        const [c, h, w] = prev.out_shape
        if (c !== cin) fail(where, `${layer.id}: kernel expects ${cin} channels, input has ${c}`)
        const want = [cout, Math.floor((h + 2 * pad - kh) / s) + 1, Math.floor((w + 2 * pad - kw) / s) + 1]
        if (!same(layer.out_shape, want)) fail(where, `${layer.id}: out_shape ${layer.out_shape} != ${want}`)
        break
      }
      case 'maxpool2d': {
        const p = layer.window!
        const s = layer.stride ?? p
        if (!prev || prev.out_shape.length !== 3) fail(where, `${layer.id}: pooling needs a 3-d input`)
        const [c, h, w] = prev.out_shape
        const want = [c, Math.floor((h - p) / s) + 1, Math.floor((w - p) / s) + 1]
        if (!same(layer.out_shape, want)) fail(where, `${layer.id}: out_shape ${layer.out_shape} != ${want}`)
        break
      }
      case 'flatten':
        if (prev && prod(prev.out_shape) !== prod(layer.out_shape)) fail(where, `${layer.id}: flatten changes the element count`)
        break
      default:
        break
    }
  }

  // runtime outputs describe exactly the layers they claim to
  const names = new Set<string>()
  for (const ro of trace.runtime_outputs) {
    if (names.has(ro.name)) fail(where, `duplicate runtime output ${ro.name}`)
    names.add(ro.name)
    const layer = byId.get(ro.layer)
    if (!layer) fail(where, `runtime output ${ro.name} refers to unknown layer ${ro.layer}`)
    if (!same(ro.shape, layer.out_shape)) fail(where, `runtime output ${ro.name} shape ${ro.shape} != layer ${layer.out_shape}`)
  }

  // parameter index lies inside the file and every referenced tensor exists
  if (trace.params) {
    const t = new Map<string, ParamTensor>()
    for (const e of trace.params.tensors) {
      if (e.offset % 4 !== 0 || e.offset + e.nbytes > trace.params.nbytes) fail(where, `tensor ${e.name} outside params file`)
      if (e.nbytes !== prod(e.shape) * 4) fail(where, `tensor ${e.name}: nbytes != 4 * size`)
      t.set(e.name, e)
    }
    for (const layer of trace.layers) {
      for (const ref of Object.values(layer.params ?? {})) if (!t.has(ref)) fail(where, `${layer.id} references missing tensor ${ref}`)
      if (layer.op === 'dense' && layer.params) {
        const w = t.get(layer.params.weight)!
        if (!same(w.shape, [layer.in_features!, layer.out_features!])) fail(where, `${layer.id}: weight shape ${w.shape}`)
      }
      if (layer.op === 'conv2d' && layer.params && !same(t.get(layer.params.kernel)!.shape, layer.kernel_shape!)) {
        fail(where, `${layer.id}: kernel tensor shape differs from kernel_shape`)
      }
    }
  } else if (trace.kind !== 'adaboost') {
    fail(where, 'model without parameter file')
  }
  if (trace.output.kind === 'vote_share' && trace.output.calibrated) fail(where, 'vote shares can never be calibrated probabilities')
}

export function parseBundleManifest(raw: unknown): BundleManifest {
  const m = raw as BundleManifest | null
  if (!m || typeof m !== 'object') throw new ContractError('not an object', 'manifest')
  if (m.schema !== BUNDLE_SCHEMA) throw new ContractError(`unsupported schema ${String(m.schema)}`, 'manifest')
  if (!m.datasets || !m.bundle_id || !m.datasets[m.default_dataset]) throw new ContractError('missing datasets / bundle_id', 'manifest')
  for (const [dsId, ds] of Object.entries(m.datasets)) {
    if (!Array.isArray(ds.classes) || ds.classes.length === 0) throw new ContractError('no class names', dsId)
    if (!ds.preprocessing || typeof ds.preprocessing.mean !== 'number') throw new ContractError('no preprocessing constants', dsId)
    for (const [id, model] of Object.entries(ds.models)) {
      const where = `${dsId}/${id}`
      if (!isModelKind(id) || !model) throw new ContractError('unknown model id', where)
      if (model.trace.kind !== id) throw new ContractError(`trace.kind ${model.trace.kind} != ${id}`, where)
      if (!model.sha256 || !model.file) throw new ContractError('missing file hash', where)
      const classes = model.trace.output.classes
      if (classes.length !== ds.classes.length || classes.some((c, i) => c !== ds.classes[i])) {
        throw new ContractError('class labels differ from the dataset', where)
      }
      validateTrace(model.trace, where)
    }
  }
  return m
}

export function datasetIds(m: BundleManifest): string[] {
  return Object.keys(m.datasets)
}

export function modelsOf(ds: DatasetEntry): ModelKind[] {
  return MODEL_ORDER.filter((k) => ds.models[k])
}

/** Layer lookup helpers used by every explorer view. */
export function layerMap(trace: Trace): Map<string, TraceLayer> {
  return new Map(trace.layers.map((l) => [l.id, l]))
}

export function layersOf(trace: Trace, op: LayerOp): TraceLayer[] {
  return trace.layers.filter((l) => l.op === op)
}

// --------------------------------------------------------------------------- training replay
export const REPLAY_SCHEMA = 'writerhand.replay/1'

/** A tensor saved in a checkpoint (row-major, float32 values as plain numbers). */
export interface ReplayTensor {
  shape: number[]
  data: number[]
  /** what part of the model this is, e.g. "W[:, :16]" when only a slice was saved */
  note?: string
}

export interface ReplaySnapshot {
  /** 0 = initial weights before any update */
  epoch: number
  metrics: { loss?: number; val_acc?: number; val_loss?: number; inner_val_loss?: number }
  /** model outputs (probabilities, or vote shares for AdaBoost) of the saved weights on `samples` */
  proba: number[][]
  tensors: Record<string, ReplayTensor>
}

/** Genuine record of one training run: nothing in it is simulated. */
export interface ReplayData {
  schema: string
  dataset: string
  model: ModelKind
  seed: number
  params: Record<string, unknown>
  /** per-epoch (or per-round for AdaBoost) measured series; all arrays have `epochs.length` entries */
  epochs: number[]
  series: Record<string, number[]>
  best_epoch: number | null
  stopped_epoch: number | null
  /** fixed training images whose predictions were recorded at every snapshot */
  samples: { indices: number[]; labels: number[]; pixels: number[][] }
  snapshots: ReplaySnapshot[]
  /** epoch whose weights were restored as the deployed model (best epoch; the last one without early stopping) */
  deployed_epoch: number
  /** true when the weights at `deployed_epoch` are bit-identical to the deployed model */
  final_matches_deployed: boolean
  note: string
}

export function parseReplay(raw: unknown, expect: { dataset: string; model: ModelKind }): ReplayData {
  const r = raw as ReplayData
  const where = `${expect.dataset}/${expect.model}/replay`
  if (!r || r.schema !== REPLAY_SCHEMA) throw new ContractError('unsupported replay schema', where)
  if (r.dataset !== expect.dataset || r.model !== expect.model) throw new ContractError('replay belongs to another model', where)
  for (const [k, v] of Object.entries(r.series)) if (v.length !== r.epochs.length) throw new ContractError(`series ${k} has ${v.length} values for ${r.epochs.length} epochs`, where)
  const n = r.samples.indices.length
  for (const s of r.snapshots) {
    if (s.proba.length !== n) throw new ContractError(`snapshot ${s.epoch}: ${s.proba.length} prediction rows for ${n} samples`, where)
    for (const [name, t] of Object.entries(s.tensors)) {
      if (t.data.length !== t.shape.reduce((a, b) => a * b, 1)) throw new ContractError(`snapshot ${s.epoch}: tensor ${name} size`, where)
    }
  }
  return r
}
