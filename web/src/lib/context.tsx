import { createContext, useContext, useSyncExternalStore } from 'react'
import type { ModelStore, StatusMap } from './modelStore'
import type { SiteData } from './types'

export const SiteDataContext = createContext<SiteData | null>(null)
export const ModelStoreContext = createContext<ModelStore | null>(null)

export function useSiteData(): SiteData {
  const v = useContext(SiteDataContext)
  if (!v) throw new Error('Site data is not available yet.')
  return v
}

export function useModelStore(): ModelStore {
  const v = useContext(ModelStoreContext)
  if (!v) throw new Error('Model store is not available yet.')
  return v
}

export function useModelStatuses(store: ModelStore): StatusMap {
  return useSyncExternalStore(store.subscribe, store.getSnapshot)
}
