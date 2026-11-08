import katex from 'katex'
import { describe, expect, it } from 'vitest'
import { EXTRA_TEX, MODEL_TEX } from '../src/pages/modelContent'

describe('equations of the Models page', () => {
  const all = [...Object.values(MODEL_TEX), ...Object.values(EXTRA_TEX)].flat(2)
  it('every line is valid TeX for KaTeX (no silent red error box)', () => {
    for (const tex of all) expect(() => katex.renderToString(tex, { displayMode: true, throwOnError: true, strict: 'error', output: 'htmlAndMathml' }), tex).not.toThrow()
  })
  it('uses an upright arg max operator, never italic letters', () => {
    for (const tex of all) expect(tex, tex).not.toMatch(/\\arg\\max/)
  })
})
