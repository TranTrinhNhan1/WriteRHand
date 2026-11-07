import type { Ort } from './onnx'

let pending: Promise<Ort> | null = null

/**
 * Loads ONNX Runtime Web on first use. The runtime and its ~14 MB WebAssembly binary are
 * served from this site (the binary is emitted as a hashed asset under the Vite base path),
 * so no CDN is involved. GitHub Pages cannot send COOP/COEP headers, hence single-threaded.
 */
export function loadOrt(): Promise<Ort> {
  pending ??= (async () => {
    const [ort, wasm] = await Promise.all([
      import('onnxruntime-web/wasm'),
      import('onnxruntime-web/ort-wasm-simd-threaded.wasm?url'),
    ])
    ort.env.wasm.numThreads = 1
    ort.env.wasm.proxy = false
    ort.env.wasm.wasmPaths = { wasm: wasm.default }
    ort.env.logLevel = 'error'
    return ort
  })().catch((err) => {
    pending = null
    throw err
  })
  return pending
}
