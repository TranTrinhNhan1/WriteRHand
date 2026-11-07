export type ModelId = 'naive_bayes' | 'softmax' | 'mlp' | 'adaboost'
export type SplitId = 'train' | 'val' | 'test'

export const MODEL_IDS: ModelId[] = ['naive_bayes', 'softmax', 'mlp', 'adaboost']
export const SPLIT_IDS: SplitId[] = ['train', 'val', 'test']

export interface Validation {
  max_abs_prob_diff?: number | null
  argmax_agreement?: number | null
  n_checked?: number | null
  ok?: boolean | null
}

export interface ManifestModel {
  title: string
  class?: string
  format: 'onnx' | 'json'
  file: string
  size_bytes?: number | null
  params?: Record<string, unknown> | null
  validation?: Validation | null
  input?: string
  output?: 'probabilities' | 'vote_shares' | string
}

export interface Preproc {
  mean: number
  std: number
  eps: number
  scale?: string
  formula?: string
  image?: string
  version?: string
}

export interface DatasetInfo {
  n_train?: number
  n_val?: number
  n_test?: number
  seed?: number
  split_fingerprint?: string
  preprocess_version?: string
}

export interface Manifest {
  name?: string
  version?: string
  created?: string
  preprocessing: Preproc
  classes: number[]
  dataset?: DatasetInfo
  models: Partial<Record<ModelId, ManifestModel>>
}

export interface SplitMetrics {
  accuracy?: number | null
  precision_macro?: number | null
  recall_macro?: number | null
  f1_macro?: number | null
  predict_time_sec?: number | null
  predict_time_ms_per_sample?: number | null
  confusion_matrix?: number[][] | null
}

export interface SklearnRef {
  class?: string
  train_time_sec?: number | null
  latency_ms_single_sample?: number | null
  splits?: Partial<Record<SplitId, { accuracy?: number | null; f1_macro?: number | null; predict_time_sec?: number | null }>>
}

export interface ResultModel {
  id: ModelId
  name: string
  class?: string
  params?: Record<string, unknown> | null
  train_time_sec?: number | null
  latency_ms_single_sample?: number | null
  splits?: Partial<Record<SplitId, SplitMetrics>>
  top_confusions?: { true: number; pred: number; count: number }[] | null
  history?: Record<string, number[]> | null
  best_epoch?: number | null
  stopped_epoch?: number | null
  in_browser?: boolean | null
  sklearn?: SklearnRef | null
}

export interface SweepPoint {
  value: unknown
  train_acc?: number | null
  val_acc?: number | null
  train_time_sec?: number | null
  stage?: string
}

export interface TuningEntry {
  best_params?: Record<string, unknown> | null
  n_trials?: number | null
  sweeps?: Record<string, SweepPoint[]> | null
  epoch_curve?: { param?: string; max?: number; chosen?: number; val_acc_at_chosen?: number } | null
  curve?: Record<string, number[]> | null
}

export interface Hardware {
  cpu?: string
  logical_cores?: number
  torch_threads?: number
  device?: string
  os?: string
  python?: string
  numpy?: string
  torch?: string
  sklearn?: string
}

export interface Results {
  generated?: string
  dataset?: DatasetInfo
  hardware?: Hardware
  git?: { commit?: string; short?: string; dirty?: boolean }
  protocol?: string
  models: ResultModel[]
  tuning?: Partial<Record<ModelId, TuningEntry>>
}

export interface SiteData {
  manifest: Manifest
  results: Results
}

export interface Prediction {
  /** Class probabilities (or vote shares), length 10, sums to 1. */
  probs: Float64Array
  /** Index of the largest entry. */
  top: number
  /** Wall-clock inference time in milliseconds. */
  ms: number
}
