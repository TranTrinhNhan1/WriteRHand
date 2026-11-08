/**
 * Real test images of a dataset (`<dataset>/test_vectors.json` of the bundle): a few images per
 * class with their true labels. They go to the models exactly as stored (pixel / 255), without the
 * drawing pipeline, which matters for datasets such as Fashion-MNIST that are hard to draw.
 */
import { useEffect, useState } from 'react'
import { sha256Hex } from '../engine/bytes'
import type { BundleManifest } from '../engine/contract'
import { CodedError, asCoded } from './errors'

export interface TestImage {
  /** index in the dataset's official test split */
  index: number
  label: number
  /** 784 values in [0, 1] */
  pixels: Float32Array
}

interface RawVectors {
  dataset?: string
  indices: number[]
  labels: number[]
  pixels: number[][]
}

/** Checks and converts the raw file; images are ordered by label, then by index. */
export function parseTestVectors(raw: unknown, nClasses: number): TestImage[] {
  const r = raw as RawVectors
  if (!r || !Array.isArray(r.indices) || !Array.isArray(r.labels) || !Array.isArray(r.pixels)) throw new Error('test vectors: missing arrays')
  const n = r.indices.length
  if (r.labels.length !== n || r.pixels.length !== n) throw new Error('test vectors: arrays differ in length')
  const out: TestImage[] = []
  for (let i = 0; i < n; i++) {
    const px = r.pixels[i]
    const label = r.labels[i]
    if (!Array.isArray(px) || px.length !== 784 || !Number.isInteger(label) || label < 0 || label >= nClasses) throw new Error(`test vectors: bad image ${i}`)
    out.push({ index: r.indices[i], label, pixels: Float32Array.from(px, (v) => Math.min(1, Math.max(0, v / 255))) })
  }
  return out.sort((a, b) => a.label - b.label || a.index - b.index)
}

const cache = new Map<string, Promise<TestImage[]>>()

export function loadTestImages(manifest: BundleManifest, baseUrl: string, dataset: string): Promise<TestImage[]> {
  const ds = manifest.datasets[dataset]
  const key = `${manifest.bundle_id}/${dataset}`
  let job = cache.get(key)
  if (!job) {
    job = (async () => {
      const file = ds.test_vectors
      const label = file.split('/').pop() ?? file
      let buf: ArrayBuffer
      try {
        const res = await fetch(`${baseUrl}/${file}`)
        if (!res.ok) throw new CodedError('err.http', { code: res.status, file: label })
        buf = await res.arrayBuffer()
      } catch (e) {
        throw e instanceof CodedError ? e : new CodedError('err.download', { file: label })
      }
      const want = manifest.files_sha256?.[file]
      if (want && (await sha256Hex(buf)) !== want) throw new CodedError('err.hash', { file: label })
      try {
        return parseTestVectors(JSON.parse(new TextDecoder().decode(buf)), ds.classes.length)
      } catch {
        throw new CodedError('err.format', { file: label })
      }
    })()
    job.catch(() => cache.delete(key))
    cache.set(key, job)
  }
  return job
}

export type TestImagesState = { state: 'idle' | 'loading' } | { state: 'ready'; images: TestImage[] } | { state: 'error'; error: CodedError }

export function useTestImages(manifest: BundleManifest, baseUrl: string, dataset: string, enabled: boolean, attempt = 0): TestImagesState {
  const [res, setRes] = useState<{ key: string; images?: TestImage[]; error?: CodedError } | null>(null)
  const key = `${dataset}#${attempt}`
  useEffect(() => {
    if (!enabled) return
    let live = true
    loadTestImages(manifest, baseUrl, dataset).then(
      (images) => live && setRes({ key, images }),
      (e: unknown) => live && setRes({ key, error: asCoded(e) }),
    )
    return () => {
      live = false
    }
  }, [manifest, baseUrl, dataset, enabled, key])
  if (!enabled) return { state: 'idle' }
  if (!res || res.key !== key) return { state: 'loading' }
  if (res.error) return { state: 'error', error: res.error }
  return { state: 'ready', images: res.images! }
}
