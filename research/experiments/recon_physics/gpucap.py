"""Run a script with the PyTorch CUDA allocator capped (shared GPU): gpucap.py [--gb 5.8] script.py [args...]"""
import os, runpy, sys
import torch

gb = 5.8
if sys.argv[1] == '--gb':
    gb = float(sys.argv[2]); del sys.argv[1:3]
tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
torch.cuda.set_per_process_memory_fraction(min(1.0, gb / tot), 0)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name='__main__')
