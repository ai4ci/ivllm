/**
 * src/semver.ts — Semantic version parsing and comparison utilities.
 *
 * Used by the backend to find the best installed vLLM version that satisfies
 * a minimum version constraint (e.g. `>= 0.19.0`).
 */

/**
 * Parse a semantic version string into numeric components.
 *
 * Handles formats like `"0.19.1"`, `"1.0.0"`, or incomplete forms like `"0.20"`
 * (treating missing parts as 0). Non-numeric parts default to 0.
 *
 * @param version — Semantic version string (e.g. `"0.19.1"`)
 * @returns `[major, minor, patch]` as numbers
 */
export function parseSemver(version: string): [number, number, number] {
    const parts = version.split('.').map((p) => parseInt(p, 10) || 0);
    // Ensure we always have 3 components
    while (parts.length < 3) parts.push(0);
    return [parts[0], parts[1], parts[2]];
}

/**
 * Compare two semantic version strings: `a < b`.
 *
 * Returns `true` if `a` is strictly less than `b`, `false` otherwise.
 *
 * @param a — First version string
 * @param b — Second version string
 * @returns true if `a < b`
 */
export function semverLt(a: string, b: string): boolean {
    const [a1, a2, a3] = parseSemver(a);
    const [b1, b2, b3] = parseSemver(b);

    if (a1 !== b1) return a1 < b1;
    if (a2 !== b2) return a2 < b2;
    return a3 < b3;
}

/**
 * Compare two semantic version strings: `a >= b`.
 *
 * Returns `true` if `a` is greater than or equal to `b`, `false` otherwise.
 *
 * @param a — First version string
 * @param b — Second version string
 * @returns true if `a >= b`
 */
export function semverGte(a: string, b: string): boolean {
    return !semverLt(a, b);
}

/**
 * Sort an array of semantic version strings in **ascending** order.
 *
 * Uses Bun's built-in version sort (`-V` flag behavior) which understands
 * dotted numeric version components.
 *
 * @param versions — Version strings to sort
 * @returns Sorted array in ascending order
 */
export function semverSort(...versions: string[]): string[] {
    return [...versions].sort((a, b) => {
        const [a1, a2, a3] = parseSemver(a);
        const [b1, b2, b3] = parseSemver(b);
        if (a1 !== b1) return a1 - b1;
        if (a2 !== b2) return a2 - b2;
        return a3 - b3;
    });
}

/**
 * Sort an array of semantic version strings in **descending** order.
 *
 * This is `semverSort` reversed — the highest version first.
 *
 * @param versions — Version strings to sort
 * @returns Sorted array in descending order
 */
export function revSemverSort(...versions: string[]): string[] {
    return semverSort(...versions).reverse();
}
