/**
 * tests/unit/semver.test.ts — Tests for semver.ts utility functions.
 *
 * Tests the semantic version comparison and sorting functions that are
 * used by `select_closest_version()` in utils.sh and `matchVllmVersion()`
 * in the TypeScript backend.
 */
import { describe, it, expect } from 'bun:test';
import { semverGte, revSemverSort } from '../../src/semver';

describe('semverGte', () => {
    it('returns true when equal', () => {
        expect(semverGte('1.0.0', '1.0.0')).toBe(true);
    });

    it('returns true when greater', () => {
        expect(semverGte('2.0.0', '1.0.0')).toBe(true);
    });

    it('returns true when minor is greater', () => {
        expect(semverGte('1.1.0', '1.0.0')).toBe(true);
    });

    it('returns true when patch is greater', () => {
        expect(semverGte('1.0.1', '1.0.0')).toBe(true);
    });

    it('returns false when less', () => {
        expect(semverGte('1.0.0', '2.0.0')).toBe(false);
    });

    it('returns false when minor is less', () => {
        expect(semverGte('1.0.0', '1.1.0')).toBe(false);
    });

    it('handles multi-digit components', () => {
        expect(semverGte('0.19.1', '0.19.0')).toBe(true);
        expect(semverGte('0.19.0', '0.19.1')).toBe(false);
        expect(semverGte('0.20.0', '0.19.99')).toBe(true);
    });
});

describe('revSemverSort', () => {
    it('sorts descending', () => {
        const result = revSemverSort('0.19.0', '0.21.0', '0.20.0', '0.18.0');
        expect(result).toEqual(['0.21.0', '0.20.0', '0.19.0', '0.18.0']);
    });

    it('handles single version', () => {
        expect(revSemverSort('1.0.0')).toEqual(['1.0.0']);
    });

    it('handles duplicates', () => {
        expect(revSemverSort('1.0.0', '1.0.0', '0.9.0')).toEqual(['1.0.0', '1.0.0', '0.9.0']);
    });

    it('sorts by major first', () => {
        expect(revSemverSort('0.99.0', '1.0.0', '0.50.0')).toEqual(['1.0.0', '0.99.0', '0.50.0']);
    });
});
