#!/bin/bash
# The queued GPU work of research/coil_segment_compression.md §10/§11, in order; every step is resumable (re-run this
# script after an interruption). Usage: run_all.sh <dir holding gpu.lock>
D=$(dirname "$0")
$D/run_diag.sh "$1"; $D/run_counts.sh "$1"; $D/run_gains.sh "$1"
