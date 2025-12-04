#!/usr/bin/env python3
import json
import sys
from pathlib import Path

build_dir = Path(sys.argv[1])
output_file = build_dir / "compile_commands.json"

commands = []

# Collect from all subproject build directories
for subdir in ["lib-build", "test-build", "server-build"]:
    compile_commands = build_dir / subdir / "compile_commands.json"
    if compile_commands.exists():
        with open(compile_commands) as f:
            commands.extend(json.load(f))

# Write merged file
with open(output_file, 'w') as f:
    json.dump(commands, f, indent=2)

print(f"Merged {len(commands)} compile commands to {output_file}")