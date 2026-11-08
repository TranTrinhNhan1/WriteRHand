/**
 * Human-readable summary of a model's computation, read only from its trace (`trace.layers` and the
 * parameter index). Nothing about depth, widths, kernels or rounds is assumed here.
 */
import type { LayerOp, Trace, TraceLayer } from '../engine/contract'

export interface LayerSummary {
  id: string
  op: LayerOp
  /** short op-specific detail, e.g. "3×3, 1→8" for a convolution */
  detail: string
  /** output shape written out, e.g. "8×26×26" */
  shape: string
  /** learned values that belong to this layer (from the parameter index), or null when unknown */
  params: number | null
}

const x = (a: readonly (number | string)[]) => a.join('×')
const size = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

function detailOf(l: TraceLayer): string {
  switch (l.op) {
    case 'dense':
      return `${l.in_features}→${l.out_features}`
    case 'conv2d': {
      const [cout, cin, kh, kw] = l.kernel_shape ?? []
      const extra = [(l.stride ?? 1) !== 1 ? `s${l.stride}` : '', (l.padding ?? 0) !== 0 ? `p${l.padding}` : ''].filter(Boolean).join(' ')
      return `${kh}×${kw}, ${cin}→${cout}${extra ? ` ${extra}` : ''}`
    }
    case 'maxpool2d':
      return `${l.window}×${l.window}${(l.stride ?? l.window) !== l.window ? ` s${l.stride}` : ''}`
    case 'gaussian_loglik':
      return `${l.n_classes}×${l.n_features}`
    case 'tree_ensemble': {
      const depth = l.depths?.length ? Math.max(...l.depths) : null
      return depth != null ? `${l.n_estimators}, ≤${depth}` : String(l.n_estimators ?? '')
    }
    default:
      return ''
  }
}

export function summarizeLayers(trace: Trace): LayerSummary[] {
  const byLayer = new Map<string, number>()
  for (const t of trace.params?.tensors ?? []) byLayer.set(t.layer, (byLayer.get(t.layer) ?? 0) + size(t.shape))
  return trace.layers.map((l) => ({
    id: l.id,
    op: l.op,
    detail: detailOf(l),
    shape: x(l.out_shape),
    params: byLayer.has(l.id) ? byLayer.get(l.id)! : null,
  }))
}

/** Total of the trace's parameter index; equals `trace.n_parameters` for every exported network. */
export function parameterCount(trace: Trace): number | null {
  if (trace.n_parameters != null) return trace.n_parameters
  if (!trace.params) return null
  return trace.params.tensors.reduce((a, t) => a + size(t.shape), 0)
}
