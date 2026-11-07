import { CodedError } from './errors'
import { DATA_BASE_URL } from './config'
import type { Manifest, Results, SiteData } from './types'

async function getJson<T>(url: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(url)
  } catch {
    throw new CodedError('fatal.reach')
  }
  if (!res.ok) throw new CodedError('fatal.http', { code: res.status })
  return (await res.json()) as T
}

/** Loads the committed, same-origin copies of manifest.json and results.json. */
export async function loadSiteData(): Promise<SiteData> {
  const [manifest, results] = await Promise.all([
    getJson<Manifest>(`${DATA_BASE_URL}/manifest.json`),
    getJson<Results>(`${DATA_BASE_URL}/results.json`),
  ])
  if (!manifest?.models || !manifest.preprocessing) throw new CodedError('fatal.shape', { file: 'manifest.json' })
  if (!Array.isArray(results?.models)) throw new CodedError('fatal.shape', { file: 'results.json' })
  return { manifest, results }
}
