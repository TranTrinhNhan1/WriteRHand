import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import * as ort from 'onnxruntime-web'
import { parseBundleManifest, type BundleManifest } from '../src/engine/contract'
import { ModelRegistry, type OrtLike } from '../src/engine/store'

/** Directory with manifest.json and one folder per dataset (not committed; fetched by CI from the pinned revision). */
export const BUNDLE_DIR = resolve(process.env.PARITY_BUNDLE_DIR ?? resolve(__dirname, '../public/bundle'))
export const hasBundle = existsSync(resolve(BUNDLE_DIR, 'manifest.json'))
/** In CI and release builds (REQUIRE_BUNDLE=1) a missing bundle is a failure, never a silent skip. */
export const bundleRequired = ['1', 'true'].includes(process.env.REQUIRE_BUNDLE ?? '')
if (bundleRequired && !hasBundle) {
  throw new Error(`REQUIRE_BUNDLE is set but no model bundle was found in ${BUNDLE_DIR}; run \`npm run bundle:fetch\``)
}

export interface TestVectors {
  dataset: string
  indices: number[]
  labels: number[]
  pixels: number[][]
  expected: Record<string, { pred: number[]; proba: number[][]; logits?: number[][] }>
}

export const readJson = <T>(rel: string): T => JSON.parse(readFileSync(resolve(BUNDLE_DIR, rel), 'utf8')) as T
export const readBytes = (rel: string): Uint8Array => new Uint8Array(readFileSync(resolve(BUNDLE_DIR, rel)))

export const loadManifest = (): BundleManifest => parseBundleManifest(readJson('manifest.json'))

/** fetch() replacement that serves files of the local bundle directory. */
export const fileFetch: typeof fetch = async (url) => {
  const rel = String(url).replace(/^.*\/bundle\//, '')
  const path = resolve(BUNDLE_DIR, rel)
  if (!existsSync(path)) return new Response('not found', { status: 404 })
  const b = readFileSync(path)
  return new Response(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer)
}

export function makeRegistry(manifest = loadManifest(), fetchImpl: typeof fetch = fileFetch) {
  return new ModelRegistry(manifest, 'http://local/bundle', { fetchImpl, ortLoader: async () => ort as unknown as OrtLike })
}

export const toPixels01 = (v: number[]) => Float32Array.from(v, (x) => x / 255)
