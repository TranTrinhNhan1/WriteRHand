const HF_BASE = 'https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST/resolve/main'

/** Where `manifest.json`, `models/*` and `test_vectors.json` of the model bundle are served from. */
export const MODEL_BASE_URL: string = (
  import.meta.env.VITE_MODEL_BASE_URL ||
  (import.meta.env.PROD ? HF_BASE : `${import.meta.env.BASE_URL}bundle`)
).replace(/\/+$/, '')

/** Same-origin copies of manifest.json and results.json (committed with the site). */
export const DATA_BASE_URL = `${import.meta.env.BASE_URL}data`

export const REPO_URL = 'https://github.com/TranTrinhNhan1/WriteRHand'
export const HF_URL = 'https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST'
