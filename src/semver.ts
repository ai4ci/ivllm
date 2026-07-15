/**
 * Compare two semantic version strings (MAJOR.MINOR.PATCH).
 *
 * Returns `true` if `a` is strictly less than `b` by lexicographic
 * major → minor → patch ordering. Components that are missing or
 * cannot be parsed as integers are treated as `0`.
 *
 * **Examples**
 *
 * ```ts
 * semverLt('0.19.0', '0.19.1') // true
 * semverLt('0.19.1', '0.19.1') // false
 * semverLt('0.19.1', '0.20.0') // true
 * semverLt('1.0.0', '0.99.0')  // false
 * ```
 * @param a - Left-hand version string (e.g. `'0.19.1'`)
 * @param b - Right-hand version string (e.g. `'0.20.0'`)
 * @returns `true` if `a < b`, otherwise `false`
 */
export function semverLt(a: string, b: string): boolean {
  const parse = (v: string) => v.split('.').map((n) => parseInt(n, 10) || 0);
  const [a1, a2, a3] = parse(a);
  const [b1, b2, b3] = parse(b);
  if (a1 !== b1) return (a1 ?? 0) < (b1 ?? 0);
  if (a2 !== b2) return (a2 ?? 0) < (b2 ?? 0);
  return (a3 ?? 0) < (b3 ?? 0);
}

/**
 * Compare two semantic version strings (MAJOR.MINOR.PATCH).
 *
 * Returns `true` if `a` is greater than or equal to `b` by lexicographic
 * major → minor → patch ordering. This is the logical inverse of
 * {@link semverLt}.
 *
 * **Examples**
 *
 * ```ts
 * semverGte('0.20.0', '0.19.1') // true
 * semverGte('0.19.1', '0.19.1') // true
 * semverGte('0.19.0', '0.19.1') // false
 * ```
 * @param a - Left-hand version string (e.g. `'0.20.0'`)
 * @param b - Right-hand version string (e.g. `'0.19.1'`)
 * @returns `true` if `a >= b`, otherwise `false`
 */
export function semverGte(a: string, b: string): boolean {
  return !semverLt(a, b);
}

/**
 * Sort an array of semantic version strings in descending order
 * (highest version first).
 *
 * Returns a **new** array; the input array is not mutated.
 *
 * **Examples**
 *
 * ```ts
 * semverSort(['0.19.0', '0.22.0', '0.19.1'])
 * // → ['0.22.0', '0.19.1', '0.19.0']
 * ```
 * @param versions - Array of version strings (e.g. `['0.19.0', '0.20.0']`)
 * @returns A new array sorted descending by version
 */
export function semverSort(versions: string[]): string[] {
  return [...versions].sort((a, b) =>
    semverLt(a, b) ? 1 : semverLt(b, a) ? -1 : 0,
  );
}

/**
 * Sort an array of semantic version strings in ascending order
 * (lowest version first).
 *
 * Returns a **new** array; the input array is not mutated.
 *
 * **Examples**
 *
 * @param versions - Array of version strings (e.g. `['0.19.0', '0.20.0']`)
 * @returns A new array sorted descending by version
 */
export function revSemverSort(versions: string[]): string[] {
  return [...versions].sort((a, b) =>
    semverLt(a, b) ? 1 : semverLt(b, a) ? -1 : 0,
  );
}
