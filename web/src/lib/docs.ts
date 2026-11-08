/**
 * The course deliverables published with the site (`public/docs/`, written by `scripts/docs.mjs prepare`).
 * `docs.json` describes each PDF; the page still links the files when the metadata is missing or malformed.
 */
export type DocId = 'report' | 'slides'

export const DOC_IDS: DocId[] = ['report', 'slides']

export interface DocFile {
  file: string
  pages: number
  bytes: number
  sha256: string
}

export type DocsMeta = Record<DocId, DocFile> & { generated: string | null }

const DOC_FILES: Record<DocId, string> = { report: 'report.pdf', slides: 'slides.pdf' }

/** URL of a published file under the site's base path, e.g. `/WriteRHand/docs/report.pdf`. */
export const docUrl = (name: string, base: string = import.meta.env.BASE_URL) => `${base.endsWith('/') ? base : `${base}/`}docs/${name}`

export const docFileUrl = (id: DocId, base?: string) => docUrl(DOC_FILES[id], base)

const isCount = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v > 0

function parseFile(v: unknown, id: DocId): DocFile | null {
  if (!v || typeof v !== 'object') return null
  const o = v as Record<string, unknown>
  if (o.file !== DOC_FILES[id] || !isCount(o.pages) || !isCount(o.bytes)) return null
  if (typeof o.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(o.sha256)) return null
  return { file: o.file, pages: o.pages, bytes: o.bytes, sha256: o.sha256 }
}

/** Validates `docs.json`; returns null when it does not have the expected shape (counts are then hidden). */
export function parseDocsMeta(json: unknown): DocsMeta | null {
  if (!json || typeof json !== 'object') return null
  const o = json as Record<string, unknown>
  const report = parseFile(o.report, 'report')
  const slides = parseFile(o.slides, 'slides')
  if (!report || !slides) return null
  return { report, slides, generated: typeof o.generated === 'string' ? o.generated : null }
}

export async function loadDocsMeta(fetcher: typeof fetch = fetch, base?: string): Promise<DocsMeta | null> {
  const res = await fetcher(docUrl('docs.json', base), { cache: 'no-cache' })
  if (!res.ok) throw new Error(`docs.json: HTTP ${res.status}`)
  return parseDocsMeta(await res.json())
}
