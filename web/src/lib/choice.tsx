import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { useBundle } from '../engine/react'
import type { ModelId } from '../engine/contract'
import { resolveDataset, resolveModel } from './selection'

/** The dataset and model the visitor picked; shared by the pages so a choice survives navigation. */
export interface Choice {
  dataset: string
  /** null only when the dataset offers no model that passed its export check */
  model: ModelId | null
  setDataset(id: string): void
  setModel(id: ModelId): void
}

const ChoiceContext = createContext<Choice | null>(null)

export function ChoiceProvider({ children }: { children: ReactNode }) {
  const { manifest } = useBundle()
  const [wantedDs, setWantedDs] = useState<string | null>(null)
  const [wantedModel, setWantedModel] = useState<ModelId | null>(null)
  const dataset = resolveDataset(manifest, wantedDs)
  const model = resolveModel(manifest.datasets[dataset], wantedModel)
  const setDataset = useCallback((id: string) => setWantedDs(id), [])
  const setModel = useCallback((id: ModelId) => setWantedModel(id), [])
  const value = useMemo(() => ({ dataset, model, setDataset, setModel }), [dataset, model, setDataset, setModel])
  return <ChoiceContext.Provider value={value}>{children}</ChoiceContext.Provider>
}

export function useChoice(): Choice {
  const v = useContext(ChoiceContext)
  if (!v) throw new Error('ChoiceProvider is missing.')
  return v
}
