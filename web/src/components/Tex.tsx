import katex from 'katex'
import { useMemo } from 'react'
import '../styles/katex.css'

interface Props {
  tex: string
  block?: boolean
}

/**
 * Typesets a TeX string with KaTeX in its Computer Modern fonts (HTML), with the MathML alongside for screen
 * readers. Input is static, trusted text.
 */
export function Tex({ tex, block }: Props) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: !!block, throwOnError: false, strict: 'ignore', trust: false, output: 'htmlAndMathml' }),
    [tex, block],
  )
  return block ? (
    <div className="tex tex--block" tabIndex={0} dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className="tex" dangerouslySetInnerHTML={{ __html: html }} />
  )
}
