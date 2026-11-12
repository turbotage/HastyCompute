"""Kernel-time split of an opspeed run profiled with nsys (research/operator_speed.md):

    nsys profile -t cuda -o <rep> opspeed op ...   then   python nsys_split.py <rep>.nsys-rep [sets_timed]

Groups the CUDA kernels of the segmented operator: gather (seg_forward_mt / seg_forward_t), scatter (seg_adjoint_mt /
seg_adjoint_t), pad (seg_pad*), crop (seg_crop*), FFT (fft_axis), other kernels; memcpy HtoD / DtoH / DtoD and memset
from the memory-operation summary. Prints seconds and shares, and writes <rep>.split.json.
"""
import csv, io, json, subprocess, sys


def stats(rep, report):
    out = subprocess.run(['nsys', 'stats', '-q', '-r', report, '-f', 'csv', rep], capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if l.strip()]
    i = next(k for k, l in enumerate(lines) if l.startswith('"Time') or l.startswith('Time'))
    return list(csv.DictReader(io.StringIO('\n'.join(lines[i:]))))


def main():
    rep = sys.argv[1]
    k = stats(rep, 'cuda_gpu_kern_sum')
    groups = {'gather (forward tile kernel)': 0.0, 'scatter (adjoint tile kernel)': 0.0, 'FFT (fft_axis)': 0.0, 'pad (S·P·x, fused)': 0.0, 'crop (Σ conj(S P) ·, fused)': 0.0, 'other kernels': 0.0}
    per = {}
    for r in k:
        name = r['Name']
        t = float(r['Total Time (ns)']) * 1e-9
        per[name[:80]] = per.get(name[:80], 0.0) + t
        if 'seg_forward' in name:
            groups['gather (forward tile kernel)'] += t
        elif 'seg_adjoint' in name:
            groups['scatter (adjoint tile kernel)'] += t
        elif 'fft_axis' in name:
            groups['FFT (fft_axis)'] += t
        elif 'seg_pad' in name:
            groups['pad (S·P·x, fused)'] += t
        elif 'seg_crop' in name:
            groups['crop (Σ conj(S P) ·, fused)'] += t
        else:
            groups['other kernels'] += t
    try:
        m = stats(rep, 'cuda_gpu_mem_time_sum')
        for r in m:
            groups['mem: ' + r['Operation'].strip('[]')] = float(r['Total Time (ns)']) * 1e-9
    except StopIteration:
        pass
    tot = sum(groups.values())
    for g, t in sorted(groups.items(), key=lambda x: -x[1]):
        print(f'{g:<36} {t:9.3f} s  {100 * t / tot:5.1f} %')
    print(f'{"total GPU time":<36} {tot:9.3f} s')
    json.dump({'groups': groups, 'kernels': per, 'total_s': tot}, open(rep.replace('.nsys-rep', '') + '.split.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
