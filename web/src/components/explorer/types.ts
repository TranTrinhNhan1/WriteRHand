import type { Execution, LoadedModel } from '../../engine/store'

/** Props of every per-model explorer view (inference mode). The execution is guaranteed fresh by the Explorer. */
export interface ExplorerViewProps {
  dataset: string
  /** display names of the classes of this dataset; index = model output index */
  classes: string[]
  loaded: LoadedModel
  /** result of the loaded model on the CURRENT input, never a stale one */
  execution: Execution
}
