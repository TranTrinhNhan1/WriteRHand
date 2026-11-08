/** The pinned model revision: `web/model-lock.json` (written by `python -m src upload --write-lock`). */
export interface ModelLock {
  repo: string
  revision: string
  bundle_id: string
  version: string
  schema: string
  manifest_sha256: string
}

const found = import.meta.glob('../../model-lock.json', { eager: true, import: 'default' }) as Record<string, ModelLock>
export const MODEL_LOCK: ModelLock | null = Object.values(found)[0] ?? null

export const hfResolveUrl = (lock: ModelLock) => `https://huggingface.co/${lock.repo}/resolve/${lock.revision}`
