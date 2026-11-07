/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BASE?: string
  readonly VITE_MODEL_BASE_URL?: string
}

declare module 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url' {
  const url: string
  export default url
}
