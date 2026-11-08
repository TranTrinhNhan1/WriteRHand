import { useEffect, useState } from 'react'

const read = (names: readonly string[]): Record<string, string> => {
  const cs = getComputedStyle(document.documentElement)
  return Object.fromEntries(names.map((n) => [n, cs.getPropertyValue(n).trim() || '#888']))
}

/** Current values of CSS custom properties (for canvas drawing); updates when the theme changes. */
export function useCssColors<T extends string>(names: readonly T[]): Record<T, string> {
  const key = names.join(',')
  const [colors, setColors] = useState(() => read(names) as Record<T, string>)
  useEffect(() => {
    const list = key.split(',')
    const update = () => setColors(read(list) as Record<T, string>)
    update()
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] })
    const mq = window.matchMedia('(prefers-color-scheme: light)')
    mq.addEventListener('change', update)
    return () => {
      mo.disconnect()
      mq.removeEventListener('change', update)
    }
  }, [key])
  return colors
}

/** Width of an element, tracked with ResizeObserver. */
export function useWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    setW(el.clientWidth)
    const ro = new ResizeObserver(() => setW(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return w
}
