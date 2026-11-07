import type * as OrtNamespace from 'onnxruntime-web'

export type Ort = typeof OrtNamespace

import { CodedError } from './errors'

export const INPUT_DIM = 784
export const N_CLASSES = 10

export interface OnnxPredictor {
  /** Runs a batch of `n` images (n * 784 floats in [0, 1]); returns n * 10 probabilities. */
  run(pixels: Float32Array, n: number): Promise<Float64Array>
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
    dispose: () => session.release(),
  }
}
