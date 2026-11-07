import katex from 'katex'
import { useMemo } from 'react'

interface Props {
  tex: string
  block?: boolean
}

/**
 * Typesets a TeX string as MathML (KaTeX does the parsing). The browser draws it with the site's own sans-serif
 * math font, so equations match the rest of the page and screen readers read real MathML. Input is static, trusted text.
 */
export function Tex({ tex, block }: Props) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: !!block, throwOnError: false, strict: 'ignore', trust: false, output: 'mathml' }),
    [tex, block],
  )
  return block ? (
    <div className="tex tex--block" tabIndex={0} dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className="tex" dangerouslySetInnerHTML={{ __html: html }} />
  )
}
