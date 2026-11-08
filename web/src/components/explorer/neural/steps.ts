import { useI18n, type Key, type Params } from '../../../i18n'
import type { Trace, TraceLayer } from '../../../engine/contract'
import { int, shape } from './format'

/** What one layer of the trace computes, as a dictionary key + parameters (for captions and screen readers). */
export function stepText(layer: TraceLayer): { key: Key; params: Params } {
  const out = shape(layer.out_shape)
  const base = { id: layer.id, out }
  switch (layer.op) {
    case 'dense':
      return { key: 'ex.shell.step.dense', params: { ...base, a: int(layer.in_features!), b: int(layer.out_features!) } }
    case 'conv2d': {
      const [cout, cin, kh, kw] = layer.kernel_shape!
      return { key: 'ex.shell.step.conv2d', params: { ...base, cout, cin, k: `${kh}×${kw}` } }
    }
    case 'maxpool2d':
      return { key: 'ex.shell.step.maxpool2d', params: { ...base, k: `${layer.window}×${layer.window}` } }
    case 'input':
    case 'flatten':
    case 'reshape':
    case 'standardize':
    case 'relu':
    case 'softmax':
      return { key: `ex.shell.step.${layer.op}`, params: base }
    default:
      return { key: 'ex.shell.step.other', params: { ...base, op: layer.op } }
  }
}

/** "Step 4 of 9: ..." for a layer-by-layer playback of `trace` at `step` (0 = nothing computed yet). */
export function useStepCaption(trace: Trace, step: number): string {
  const { t } = useI18n()
  const n = trace.layers.length
  if (step <= 0) return t('ex.shell.step.start', { n })
  const s = stepText(trace.layers[Math.min(n, step) - 1])
  return `${t('ex.shell.step.of', { k: step, n })}: ${t(s.key, s.params)}`
}
