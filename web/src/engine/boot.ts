/** Loads the pinned bundle manifest (and per-dataset results) before the app renders. */
import { CodedError } from '../lib/errors'
import { sha256Hex } from './bytes'
import { parseBundleManifest, ContractError, type BundleManifest } from './contract'
import { hfResolveUrl, MODEL_LOCK, type ModelLock } from './lock'
import type { Results } from '../lib/types'
import { parseExtraModels, type ExtraResults } from '../lib/extraModels'

export interface Booted {
  manifest: BundleManifest
  /** measured numbers per dataset (dashboard) */
  results: Record<string, Results>
  /** measured models that are not in the bundle (results pages only), per dataset */
  extra: Record<string, ExtraResults>
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

/** `fresh`: same-origin site data is revalidated on every load (GitHub Pages caches for 10 min, which
 * would otherwise let a browser pair a new script with the previous deploy's manifest). */
async function getBuf(url: string, fresh = false): Promise<ArrayBuffer> {
  let res: Response
  try {
    res = await fetch(url, fresh ? { cache: 'no-cache' } : undefined)
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
  const buf = await getBuf(fromBundle ? `${baseUrl}/manifest.json` : `${dataBase}/manifest.json`, true)
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
      const b = await getBuf(`${fromBundle ? baseUrl : dataBase}/${ds.results}`, true)
      results[id] = JSON.parse(new TextDecoder().decode(b)) as Results
    }),
  )
  // committed with the site next to the manifest; it never changes which models run in the browser
  let extra: Record<string, ExtraResults> = {}
  try {
    extra = parseExtraModels(JSON.parse(new TextDecoder().decode(await getBuf(`${dataBase}/extra_models.json`, true))))
  } catch {
    extra = {}
  }
  return { manifest, results, extra, baseUrl, lock }
}
