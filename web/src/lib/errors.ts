import type { Key, Params } from '../i18n'

/** An error whose text is a dictionary key, so it can be shown in whichever language is active. */
export class CodedError extends Error {
  constructor(
    readonly key: Key,
    readonly params?: Params,
  ) {
    super(key)
    this.name = 'CodedError'
  }
}

export function asCoded(err: unknown, fallback: Key = 'err.generic'): CodedError {
  return err instanceof CodedError ? err : new CodedError(fallback)
}
