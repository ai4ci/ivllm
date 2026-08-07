#!/bin/bash

if [[ -e "$PROJECTDIR/engine/vllm/0.25.1/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py.orig" ]]; then
  mv \
    $PROJECTDIR/engine/vllm/0.25.1/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py.orig \
    $PROJECTDIR/engine/vllm/0.25.1/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py
fi

python3 - <<'EOF'
import pathlib

path = pathlib.Path(
    "/projects/b6ax/engine/vllm/0.25.1/lib/python3.12/site-packages/"
    "vllm/distributed/device_communicators/flashinfer_all_reduce.py"
)
text = path.read_text()

old_import = "from vllm.distributed.parallel_state import get_node_count"
new_import = "from vllm.distributed.parallel_state import _node_count, get_node_count, get_tp_group"
assert text.count(old_import) == 1, "import line not found/not unique — aborting, no changes made"

old_guard = 'if get_node_count() > 1 and backend == "trtllm":'
new_guard = 'if _node_count(get_tp_group().cpu_group) > 1 and backend == "trtllm":'
assert text.count(old_guard) == 1, "guard line not found/not unique — aborting, no changes made"

backup = path.with_suffix(path.suffix + ".orig")
if not backup.exists():
    backup.write_text(path.read_text())
    print("backup written:", backup)
else:
    print("backup already exists, leaving it alone:", backup)

path.write_text(text.replace(old_import, new_import).replace(old_guard, new_guard))
print("patched:", path)
EOF
