import { describe, expect, it } from 'vitest'
import { DICTS, translate, translateList, type Key, type ListKey } from '../src/i18n'
import { en } from '../src/i18n/en'
import { vi } from '../src/i18n/vi'

/** Flattens nested objects to dotted paths. Arrays become `path[]` leaves, with their length recorded separately. */
function flatten(value: unknown, prefix = '', out: Record<string, string | string[]> = {}): Record<string, string | string[]> {
  if (typeof value === 'string') {
    out[prefix] = value
  } else if (Array.isArray(value)) {
    out[prefix] = value as string[]
  } else if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${k}` : k, out)
  }
  return out
}

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

describe('dictionaries', () => {
  const a = flatten(en)
  const b = flatten(vi)

  it('have exactly the same keys in English and Vietnamese', () => {
    expect(Object.keys(b).sort()).toEqual(Object.keys(a).sort())
  })

  it('have lists of the same length', () => {
    for (const [k, v] of Object.entries(a)) {
      if (Array.isArray(v)) expect((b[k] as string[]).length, k).toBe(v.length)
    }
  })

  it('use the same {placeholders} in both languages', () => {
    for (const [k, v] of Object.entries(a)) {
      const pairs: [string, string][] = Array.isArray(v) ? v.map((s, i) => [s, (b[k] as string[])[i]]) : [[v, b[k] as string]]
      for (const [x, y] of pairs) expect(placeholders(y), k).toEqual(placeholders(x))
    }
  })

  it('have no empty strings except the optional ones', () => {
    for (const lang of ['en', 'vi'] as const) {
      for (const [k, v] of Object.entries(flatten(DICTS[lang]))) {
        const items = Array.isArray(v) ? v : [v]
        for (const item of items) expect(item.trim().length, `${lang}:${k}`).toBeGreaterThan(0)
      }
    }
  })

  it('avoid the "from X to Y" construction the copy rules ban', () => {
    for (const lang of ['en', 'vi'] as const) {
      for (const [k, v] of Object.entries(flatten(DICTS[lang]))) {
        const text = (Array.isArray(v) ? v : [v]).join(' ')
        expect(/\bfrom\b[^.]{1,60}\bto\b/i.test(text), `${lang}:${k}`).toBe(false)
        expect(/\btừ\b[^.]{1,60}\bđến\b/i.test(text), `${lang}:${k}`).toBe(false)
      }
    }
  })
})

describe('translate', () => {
  it('fills placeholders and falls back to the key', () => {
    expect(translate('en', 'rec.samples.draw', { digit: 7 })).toBe('Draw a 7')
    expect(translate('vi', 'rec.samples.draw', { digit: 7 })).toBe('Vẽ số 7')
    expect(translate('en', 'nope.missing' as Key)).toBe('nope.missing')
  })

  it('returns lists', () => {
    expect(translateList('en', 'nope' as ListKey)).toEqual([])
  })
})
