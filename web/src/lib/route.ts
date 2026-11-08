import { useEffect, useState } from 'react'

/*
 * Hash routing (`#/models`): the part after `#` is never sent to the server, so GitHub Pages always serves the
 * site's index.html and reloading or sharing a link to any page works without a 404.html fallback.
 */

export type Route = 'recognise' | 'models' | 'results' | 'about' | 'documents'

export const ROUTES: { id: Route; hash: string }[] = [
  { id: 'recognise', hash: '#/' },
  { id: 'models', hash: '#/models' },
  { id: 'results', hash: '#/results' },
  { id: 'about', hash: '#/about' },
  { id: 'documents', hash: '#/documents' },
]

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#\/?/, '').split(/[?/]/)[0]
  return ROUTES.find((r) => r.hash === `#/${path}`)?.id ?? 'recognise'
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash))
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}
