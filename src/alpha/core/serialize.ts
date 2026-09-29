/**
 * Alpha Core — weight serialisation.
 *
 * Checkpoints store float32 buffers as base64 so a full model stays small
 * enough to persist in Alpha's own store (and to commit as a fixture in tests)
 * without pulling in a binary file format dependency.
 */

export function float32ToBase64(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return base64Encode(binary);
}

export function base64ToFloat32(encoded: string): Float32Array {
  const binary = base64Decode(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length / 4);
}

function base64Encode(binary: string): string {
  const g = globalThis as { btoa?: (s: string) => string };
  if (typeof g.btoa === "function") return g.btoa(binary);
  return Buffer.from(binary, "binary").toString("base64");
}

function base64Decode(encoded: string): string {
  const g = globalThis as { atob?: (s: string) => string };
  if (typeof g.atob === "function") return g.atob(encoded);
  return Buffer.from(encoded, "base64").toString("binary");
}

/** Rough byte size of a checkpoint payload, for reporting to the UI. */
export function base64ByteLength(encoded: string): number {
  return Math.floor((encoded.length * 3) / 4);
}
