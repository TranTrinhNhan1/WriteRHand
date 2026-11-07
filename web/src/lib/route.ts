import { useEffect, useState } from 'react'

export type Route = 'recognise' | 'models' | 'results' | 'about'

export const ROUTES: { id: Route; hash: string }[] = [
  { id: 'recognise', hash: '#/' },
  { id: 'models', hash: '#/models' },
  { id: 'results', hash: '#/results' },
  { id: 'about', hash: '#/about' },
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
