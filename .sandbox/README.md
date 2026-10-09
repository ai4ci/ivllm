# .sandbox

Configuration for scoder, which runs tools in a bubblewrap sandbox.

This whole directory is bound **read-only** inside the sandbox, so a tool
running in there can read these files but cannot change what it is allowed to
do. Edit them from outside.

- `readonly`  — paths kept read-only inside the sandbox
- `readwrite` — `$HOME` directories opened read-write
- `ports`     — localhost ports forwarded into the sandbox

scoder creates this directory and these files with sane defaults the first
time it runs in a project. Each file documents itself in its own comments.
