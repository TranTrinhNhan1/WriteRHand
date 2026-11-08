import type { ModelKind } from '../engine/contract'

export type { ModelKind }
/** Models measured in results/ but not exported to the browser bundle (no ONNX, no trace). */
export type ExtraModelId = 'svm' | 'stacking'
export const EXTRA_ORDER: ExtraModelId[] = ['svm', 'stacking']
/** Every model of the results pages: the five browser models first, then the measured-only ones. */
export type ResultId = ModelKind | ExtraModelId
export type SplitId = 'train' | 'val' | 'test'

export const SPLIT_IDS: SplitId[] = ['train', 'val', 'test']

export interface DatasetInfo {
  n_train?: number
  n_val?: number
  n_test?: number
  seed?: number
  split_fingerprint?: string
  preprocess_version?: string
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

/** Hold-out stacking: what the meta-learner was fitted on and which base models it combines. */
export interface StackingInfo {
  /** the split the meta-learner was fitted on: its accuracy on that split (and on train) is not a clean estimate */
  fitted_on: 'val'
  meta_learner?: string
  n_features?: number | null
  meta_fit_sec?: number | null
  /** sum of the base models' training times (train_time_sec of the entry = this + meta_fit_sec) */
  base_fit_sec?: number | null
  selected_l2?: number | null
  cv_val_acc?: number | null
  soft_vote_test_accuracy?: number | null
  bases: { id: string; test_accuracy?: number | null; weight?: number | null }[]
}

export interface ResultModel {
  id: ResultId
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
  stacking?: StackingInfo | null
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

// ---------------------------------------------------------------- augmentation study
/** Strength of each transformation: a number, or a [lo, hi] range for scaling. */
export type AugStrengths = Record<string, number | [number, number] | number[]>

export interface AugRun {
  seed: number
  n_train: number
  fit_time_sec: number
  train_acc: number
  val_acc: number
  test_acc: number
  val_f1_macro: number
  test_f1_macro: number
  robust_test_acc: Record<string, number>
  stopped_epoch?: number
}

export interface AugCondition {
  ops: string[]
  runs: AugRun[]
  train_acc_mean: number
  train_acc_std: number
  val_acc_mean: number
  val_acc_std: number
  test_acc_mean: number
  test_acc_std: number
  robust_test_acc_mean: Record<string, number>
  val_delta_vs_baseline_mean?: number
  val_delta_vs_baseline_sd?: number
  augmentation_time_sec: number
}

/** What `src/experiments/augment.py::run_ablation` writes. */
export interface AugStudy {
  schema: number
  model: string
  params: Record<string, unknown>
  k_copies: number
  seeds: number[]
  defaults: AugStrengths
  stress: AugStrengths
  selection_rule: string
  selected_by_validation: string
  conditions: Record<string, AugCondition>
}

// ---------------------------------------------------------------- CNN vs MLP
export interface CnnVsMlpRow {
  id: 'mlp' | 'cnn'
  n_parameters: number
  onnx_bytes: number
  train_time_sec: number
  latency_ms: number
  predict_time_10k_sec: number
  epochs: number | null
  accuracy: { train: number; val: number; test: number }
  precision_macro: number
  recall_macro: number
  f1_macro: number
  generalization_gap: number
  robust: Record<string, number>
}

export interface CnnVsMlp {
  rows: CnnVsMlpRow[]
  mcnemar?: { only_mlp: number; only_cnn: number; p: number }
  interpretability_note?: string
}

/** A model trained on dataset `trained_on`, evaluated on the test set of the dataset whose results hold this entry. */
export interface CrossEval {
  trained_on: string
  rows: { model: string; accuracy: number }[]
  note: string
}

export interface Results {
  generated?: string
  dataset?: DatasetInfo
  hardware?: Hardware
  git?: { commit?: string; short?: string; dirty?: boolean }
  protocol?: string
  models: ResultModel[]
  tuning?: Partial<Record<ResultId, TuningEntry>>
  augmentation?: { mlp?: AugStudy; cnn?: AugStudy }
  cnn_vs_mlp?: CnnVsMlp
  cross_eval?: CrossEval[]
}
