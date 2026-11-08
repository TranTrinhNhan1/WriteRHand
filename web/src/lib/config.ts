import type { ModelLock } from '../engine/lock'

export const REPO_URL = 'https://github.com/TranTrinhNhan1/WriteRHand'

/** Hugging Face repository of the model bundle when no lock file pins one (local development). */
const DEFAULT_HF_REPO = 'TranTrinhNhan1111/WriteRHand-MNIST'

export const hfRepoUrl = (lock: ModelLock | null) => `https://huggingface.co/${lock?.repo ?? DEFAULT_HF_REPO}`

/** The exact pinned revision of the model files, or null without a lock. */
export const hfRevisionUrl = (lock: ModelLock | null) => (lock ? `https://huggingface.co/${lock.repo}/tree/${lock.revision}` : null)

export const shortHash = (s: string | null | undefined, n = 8) => (s ? s.slice(0, n) : '')
