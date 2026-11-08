import { describe, expect, it } from 'vitest'
import { docFileUrl, docUrl, loadDocsMeta, parseDocsMeta } from '../src/lib/docs'
import { bytes } from '../src/lib/format'

const SHA = 'a'.repeat(64)
const good = {
  report: { file: 'report.pdf', pages: 26, bytes: 1217491, sha256: SHA },
  slides: { file: 'slides.pdf', pages: 20, bytes: 1874787, sha256: SHA },
  generated: '2026-11-08T11:12:20.343Z',
}

describe('docs metadata', () => {
  it('builds URLs under the base path', () => {
    expect(docUrl('docs.json', '/WriteRHand/')).toBe('/WriteRHand/docs/docs.json')
    expect(docUrl('docs.json', '/WriteRHand')).toBe('/WriteRHand/docs/docs.json')
    expect(docFileUrl('report', '/')).toBe('/docs/report.pdf')
    expect(docFileUrl('slides', '/WriteRHand/')).toBe('/WriteRHand/docs/slides.pdf')
  })

  it('accepts the shape written by scripts/docs.mjs', () => {
    expect(parseDocsMeta(good)).toEqual(good)
    expect(parseDocsMeta({ ...good, generated: undefined })?.generated).toBeNull()
  })

  it('rejects missing or invented counts, wrong files and bad hashes', () => {
    expect(parseDocsMeta(null)).toBeNull()
    expect(parseDocsMeta('x')).toBeNull()
    expect(parseDocsMeta({ report: good.report })).toBeNull()
    expect(parseDocsMeta({ ...good, report: { ...good.report, pages: 0 } })).toBeNull()
    expect(parseDocsMeta({ ...good, report: { ...good.report, pages: 2.5 } })).toBeNull()
    expect(parseDocsMeta({ ...good, slides: { ...good.slides, pages: '20' } })).toBeNull()
    expect(parseDocsMeta({ ...good, slides: { ...good.slides, file: 'report.pdf' } })).toBeNull()
    expect(parseDocsMeta({ ...good, slides: { ...good.slides, sha256: 'abc' } })).toBeNull()
    expect(parseDocsMeta({ ...good, report: { ...good.report, bytes: -1 } })).toBeNull()
  })

  it('loads docs.json, and fails on an HTTP error instead of inventing numbers', async () => {
    const ok = (async () => new Response(JSON.stringify(good), { status: 200 })) as typeof fetch
    expect(await loadDocsMeta(ok, '/')).toEqual(good)
    const missing = (async () => new Response('nope', { status: 404 })) as typeof fetch
    await expect(loadDocsMeta(missing, '/')).rejects.toThrow(/404/)
  })

  it('formats the file sizes shown on the cards', () => {
    expect(bytes(good.report.bytes)).toBe('1.2 MB')
    expect(bytes(good.slides.bytes)).toBe('1.8 MB')
  })
})
