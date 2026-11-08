import type { Trace } from './contract'
import { sha256Hex } from './bytes'

/** Learned parameters of one model: named float32 views over a single downloaded buffer. */
export class ParamSet {
  private readonly views = new Map<string, { data: Float32Array; shape: number[] }>()

  constructor(trace: Trace, buf: ArrayBuffer) {
    const p = trace.params
    if (!p) throw new Error('model has no parameter file')
    if (buf.byteLength !== p.nbytes) throw new Error(`parameter file is ${buf.byteLength} bytes, manifest says ${p.nbytes}`)
    for (const t of p.tensors) {
      this.views.set(t.name, { data: new Float32Array(buf, t.offset, t.nbytes / 4), shape: t.shape })
    }
  }

  get(name: string): Float32Array {
    const v = this.views.get(name)
    if (!v) throw new Error(`unknown parameter tensor ${name}`)
    return v.data
  }

  shape(name: string): number[] {
    return this.views.get(name)!.shape
  }

  names(): string[] {
    return [...this.views.keys()]
  }
}

export async function loadParams(trace: Trace, buf: ArrayBuffer, verify = true): Promise<ParamSet> {
  if (verify && trace.params && (await sha256Hex(buf)) !== trace.params.sha256) {
    throw new Error('parameter file does not match the hash in the manifest')
  }
  return new ParamSet(trace, buf)
}
