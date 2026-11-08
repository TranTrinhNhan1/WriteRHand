import { describe, expect, it } from 'vitest'
import { activeLang, translate } from '../src/i18n'
import { cellOpacity, linearTicks, logTicks, makeScale, tickLabel } from '../src/lib/scale'
import { paramValue } from '../src/lib/format'

describe('scale helpers', () => {
  it('produces round linear ticks covering the range', () => {
    const t = linearTicks(0.83, 0.98, 5)
    expect(t[0]).toBeLessThanOrEqual(0.83)
    expect(t[t.length - 1]).toBeGreaterThanOrEqual(0.98)
    expect(t.length).toBeGreaterThan(2)
  })

  it('produces decade ticks for log axes', () => {
    expect(logTicks(0.003, 40)).toEqual([0.001, 0.01, 0.1, 1, 10, 100])
    expect(logTicks(5, 5).length).toBe(2)
  })

  it('maps values for linear and log scales', () => {
    expect(makeScale('linear', [0, 10], [0, 100])(2.5)).toBe(25)
    expect(makeScale('log', [1, 1000], [0, 300])(10)).toBeCloseTo(100, 6)
  })

  it('formats ticks', () => {
    expect(tickLabel(0.001)).toBe('0.001')
    expect(tickLabel(1e-5)).toBe('1e−5')
    expect(tickLabel(100)).toBe('100')
  })

  it('keeps cell opacity within [0, 1]', () => {
    expect(cellOpacity(-1)).toBe(0)
    expect(cellOpacity(2)).toBe(1)
    expect(cellOpacity(0.5)).toBeGreaterThan(0.5)
  })

  it('labels hyper-parameter values', () => {
    expect(paramValue([128, 64])).toBe('[128, 64]')
    expect(paramValue([])).toBe(translate(activeLang(), 'common.none'))
    expect(paramValue(0.001)).toBe('0.001')
    expect(paramValue(1e-5)).toBe('1e−5')
    expect(paramValue(null)).toBe(translate(activeLang(), 'common.none'))
  })
})

describe('number formatting', () => {
  it('groups five or more digits with a narrow space and never uses commas', async () => {
    const { int } = await import('../src/lib/format')
    expect(int(60000)).toBe('60 000')
    expect(int(9999)).toBe('9999')
    expect(int(1234567)).toBe('1 234 567')
    expect(int(null)).toBe('–')
  })
})
