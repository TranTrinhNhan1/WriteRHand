/** SHA-256 (hex) of a buffer; uses Web Crypto, available in browsers and Node 20+. */
export async function sha256Hex(buf: ArrayBuffer | Uint8Array): Promise<string> {
  const data = buf instanceof Uint8Array ? buf : new Uint8Array(buf)
  const digest = await crypto.subtle.digest('SHA-256', data as BufferSource)
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Cheap 32-bit FNV-1a fingerprint of a float array, used to tell inputs apart (stale-result guard). */
export function fingerprint(a: ArrayLike<number>): string {
  let h = 0x811c9dc5
  for (let i = 0; i < a.length; i++) {
    const v = Math.round(a[i] * 65535)
    h ^= v & 0xff
    h = Math.imul(h, 0x01000193)
    h ^= (v >> 8) & 0xff
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16).padStart(8, '0')
}
