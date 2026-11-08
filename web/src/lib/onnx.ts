import type * as OrtNamespace from 'onnxruntime-web'

export type Ort = typeof OrtNamespace

import { CodedError } from './errors'

export const INPUT_DIM = 784
export const N_CLASSES = 10

export interface OnnxPredictor {
  /** Runs a batch of `n` images (n * 784 floats in [0, 1]); returns n * 10 probabilities. */
  run(pixels: Float32Array, n: number): Promise<Float64Array>
  /** Runs one image and returns the named extra outputs, or null when this model file does not have them. */
  trace(pixels: Float32Array, names: string[]): Promise<Record<string, Float32Array> | null>
  dispose(): Promise<void>
}

export async function createOnnxPredictor(
  ort: Ort,
  bytes: Uint8Array,
  inputName = 'pixels',
  outputName = 'probabilities',
): Promise<OnnxPredictor> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  if (!session.inputNames.includes(inputName) || !session.outputNames.includes(outputName)) {
    await session.release()
    throw new CodedError('err.signature')
  }
  return {
    async run(pixels, n) {
      const input = new ort.Tensor('float32', pixels, [n, INPUT_DIM])
      const out = await session.run({ [inputName]: input })
      const data = out[outputName].data as Float32Array
      if (data.length !== n * N_CLASSES) throw new CodedError('err.size')
      return Float64Array.from(data)
    },
    async trace(pixels, names) {
      if (!names.every((n) => session.outputNames.includes(n))) return null
      const input = new ort.Tensor('float32', pixels, [1, INPUT_DIM])
      const out = await session.run({ [inputName]: input }, names)
      const res: Record<string, Float32Array> = {}
      for (const n of names) res[n] = Float32Array.from(out[n].data as Float32Array)
      return res
    },
    dispose: () => session.release(),
  }
}
