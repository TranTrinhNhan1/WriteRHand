import type { ReactNode } from 'react'

/** A Results-page section: sticky heading and intro on the left, content on the right (wide screens). */
export function Section({ id, title, children, intro }: { id: string; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <section className="rsection" aria-labelledby={id}>
      <div className="rsection__head">
        <h2 id={id} className="h2">
          {title}
        </h2>
        <div className="rsection__intro">{intro}</div>
      </div>
      <div className="rsection__body">{children}</div>
    </section>
  )
}
