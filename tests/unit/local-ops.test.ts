/**
 * tests/unit/local-ops.test.ts — Tests for local-ops.ts helper functions.
 *
 * Tests the local (non-SSH) operations that run on the client machine:
 * port checking, health checking, and model catalog queries.
 */
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { isLocalPortInUse, isHealthy, queryModels } from '../../src/local-ops';
import http from 'http';

describe('isLocalPortInUse', () => {
    let server: http.Server | null = null;
    let port = 0;

    beforeEach(() => {
        port = 59000 + Math.floor(Math.random() * 5000);
    });

    afterEach(() => {
        if (server) {
            server.close();
            server = null;
        }
    });

    it('returns null for unused port', async () => {
        const result = await isLocalPortInUse(port);
        expect(result).toBeNull();
    });

    it('detects used port', async () => {
        server = http.createServer((req, res) => {
            res.writeHead(200);
            res.end('ok');
        });
        await new Promise<void>((resolve) => {
            server!.listen(port, () => resolve());
        });

        const result = await isLocalPortInUse(port);
        expect(result).not.toBeNull();
        expect(result!.pid).toBeDefined();
        expect(typeof result!.process).toBe('string');
    });
});

describe('isHealthy', () => {
    let server: http.Server | null = null;
    let port = 0;

    beforeEach(() => {
        port = 59000 + Math.floor(Math.random() * 5000);
    });

    afterEach(() => {
        if (server) {
            server.close();
            server = null;
        }
    });

    it('returns true for healthy server', async () => {
        server = http.createServer((req, res) => {
            if (req.url === '/health') {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end('{"status":"ok"}');
            } else {
                res.writeHead(404);
                res.end('not found');
            }
        });
        await new Promise<void>((resolve) => {
            server!.listen(port, () => resolve());
        });

        const result = await isHealthy(port, 2000);
        expect(result).toBe(true);
    });

    it('returns false for server without /health endpoint', async () => {
        server = http.createServer((req, res) => {
            // Never respond to /health
            res.writeHead(404);
            res.end('not found');
        });
        await new Promise<void>((resolve) => {
            server!.listen(port, () => resolve());
        });

        const result = await isHealthy(port, 2000);
        expect(result).toBe(false);
    });

    it('returns false for timeout', async () => {
        const result = await isHealthy(59996, 100);
        expect(result).toBe(false);
    });
});

describe('queryModels', () => {
    let server: http.Server | null = null;
    let port = 0;

    beforeEach(() => {
        port = 59000 + Math.floor(Math.random() * 5000);
    });

    afterEach(() => {
        if (server) {
            server.close();
            server = null;
        }
    });

    it('returns model list from /v1/models', async () => {
        server = http.createServer((req, res) => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(
                JSON.stringify({
                    object: 'list',
                    data: [{ id: 'test-model', max_model_len: 4096 }],
                }),
            );
        });
        await new Promise<void>((resolve) => {
            server!.listen(port, () => resolve());
        });

        const result = await queryModels(port, 2000);
        expect(result.object).toBe('list');
        expect(result.data).toHaveLength(1);
        expect(result.data[0]?.id).toBe('test-model');
        expect(result.data[0]?.max_model_len).toBe(4096);
    });

    it('throws on non-2xx response', async () => {
        server = http.createServer((req, res) => {
            res.writeHead(500);
            res.end('error');
        });
        await new Promise<void>((resolve) => {
            server!.listen(port, () => resolve());
        });

        expect(queryModels(port, 2000)).rejects.toThrow();
    });
});
