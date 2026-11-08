/** Loads the pinned bundle manifest (and per-dataset results) before the app renders. */
import { CodedError } from '../lib/errors'
import { sha256Hex } from './bytes'
import { parseBundleManifest, ContractError, type BundleManifest } from './contract'
import { hfResolveUrl, MODEL_LOCK, type ModelLock } from './lock'
import type { Results } from '../lib/types'

export interface Booted {
  manifest: BundleManifest
  /** measured numbers per dataset (dashboard) */
  results: Record<string, Results>
  /** where model files are fetched from: the pinned Hugging Face revision, or a local bundle in development */
  baseUrl: string
  lock: ModelLock | null
}

const trim = (s: string) => s.replace(/\/+$/, '')

export function modelBaseUrl(lock: ModelLock | null = MODEL_LOCK): string {
  const override = import.meta.env.VITE_MODEL_BASE_URL as string | undefined
  if (override) return trim(override)
  if (lock && import.meta.env.PROD) return hfResolveUrl(lock)
  return `${import.meta.env.BASE_URL}bundle`
}

async function getBuf(url: string): Promise<ArrayBuffer> {
  let res: Response
  try {
    res = await fetch(url)
  } catch {
    throw new CodedError('fatal.reach')
  }
  if (!res.ok) throw new CodedError('fatal.http', { code: res.status })
  return res.arrayBuffer()
}

/**
 * The manifest comes from the same origin (committed with the site) and must be byte-identical
 * to the one in the pinned model revision: its SHA-256 is compared with `model-lock.json`.
 */
export async function loadBundle(): Promise<Booted> {
  const lock = MODEL_LOCK
  const dataBase = `${import.meta.env.BASE_URL}data`
  const baseUrl = modelBaseUrl(lock)
  // production: the manifest committed with the site (checked against the lock below);
  // development: the local bundle itself, so a stale committed file can never be mixed with new models
  const fromBundle = !import.meta.env.PROD
  const buf = await getBuf(fromBundle ? `${baseUrl}/manifest.json` : `${dataBase}/manifest.json`)
  if (lock && import.meta.env.PROD && (await sha256Hex(buf)) !== lock.manifest_sha256) {
    throw new CodedError('fatal.pin', { file: 'manifest.json' })
  }
  let manifest: BundleManifest
  try {
    manifest = parseBundleManifest(JSON.parse(new TextDecoder().decode(buf)))
  } catch (e) {
    throw e instanceof ContractError ? new CodedError('err.contract', { file: e.where }) : new CodedError('fatal.shape', { file: 'manifest.json' })
  }
  if (lock && import.meta.env.PROD && manifest.bundle_id !== lock.bundle_id) throw new CodedError('fatal.pin', { file: 'manifest.json' })
  const results: Record<string, Results> = {}
  await Promise.all(
    Object.entries(manifest.datasets).map(async ([id, ds]) => {
      const b = await getBuf(`${fromBundle ? baseUrl : dataBase}/${ds.results}`)
      results[id] = JSON.parse(new TextDecoder().decode(b)) as Results
    }),
  )
  return { manifest, results, baseUrl, lock }
}
