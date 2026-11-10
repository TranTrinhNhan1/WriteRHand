import { createContext, useContext, type ReactNode } from 'react'

/** Number of each Results section on the current tab, by heading id; sections outside a provider are not numbered. */
export const SectionNumbers = createContext<Record<string, number>>({})

/** A Results-page section: sticky heading and intro on the left, content on the right (wide screens). */
export function Section({ id, title, children, intro }: { id: string; title: string; intro: ReactNode; children: ReactNode }) {
  const n = useContext(SectionNumbers)[id]
  return (
    <section className="rsection" aria-labelledby={id}>
      <div className="rsection__head">
        <h2 id={id} className="h2" tabIndex={-1}>
          {n != null && (
            <span className="rsection__num" aria-hidden="true">
              {n}
            </span>
          )}
          {title}
        </h2>
        <div className="rsection__intro">{intro}</div>
      </div>
      <div className="rsection__body">{children}</div>
    </section>
  )
}
