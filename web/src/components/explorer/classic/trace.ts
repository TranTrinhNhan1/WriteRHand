import type { RuntimeRole, Trace, TraceLayer } from '../../../engine/contract'
import type { Execution } from '../../../engine/store'

/** Height and width of the input image, read from the trace's input layer. */
export function imageShape(trace: Trace): { h: number; w: number } {
  const input = trace.layers.find((l) => l.op === 'input')
  const s = input?.out_shape ?? []
  if (s.length >= 2) return { h: s[s.length - 2], w: s[s.length - 1] }
  const n = s[0] ?? 0
  const side = Math.round(Math.sqrt(n))
  return { h: side, w: side }
}

export function layerOf(trace: Trace, op: TraceLayer['op']): TraceLayer {
  const l = trace.layers.find((x) => x.op === op)
  if (!l) throw new Error(`trace has no ${op} layer`)
  return l
}

/** The runtime tensor with a given role (e.g. the standardised input), as returned for THIS execution. */
export function runtimeOf(trace: Trace, exec: Execution, role: RuntimeRole): Float32Array {
  const ro = trace.runtime_outputs.find((r) => r.role === role)
  const v = ro ? exec.outputs[ro.name] : undefined
  if (!v) throw new Error(`execution has no ${role} output`)
  return v
}
