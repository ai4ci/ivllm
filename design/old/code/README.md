# v2 Reference Code

This directory contains copies of key v2 source files at the `v2-final` git tag.
They are here for convenient browsing during the v3 migration.

**These are snapshots and will go stale.** The authoritative reference is:

```bash
git show v2-final:src/templates/inference.ts
git diff v2-final  # all changes since checkpoint
```

**Files:**

| File | v2 purpose | v3 fate |
|------|-----------|---------|
| `inference.ts` | 1,176-line bash template generator | Replaced by `lib/utils.sh` + `lib/preamble.sh` |
| `session-helper.ts` | Session lifecycle pipeline | Logic distributed to `connect.ts` + bash framework |
| `monitors.ts` | LOCAL-side polling + heartbeat | Replaced by bash `monitor_*` functions |
| `start.ts` | `ivllm start` command | Replaced by `connect.ts` |
| `interactive.ts` | `ivllm interactive` command | Replaced by `connect.ts` |
| `stop.ts` | `ivllm stop` command | Replaced by `cancel.ts` |

**These copies will be removed when the v3 migration is complete** (Phase M7).
