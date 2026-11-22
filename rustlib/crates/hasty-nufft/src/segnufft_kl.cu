// Knot-path layout of the segmented NUFFT (research/sample_kernels.md §17; host segnufft/kl.rs). Standalone NVRTC
// module. Defines: TT (tile edge, 4), NCH (grid channels = coils per batch × segments).
struct cx { float x, y; };
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };

// Zero the zero-count tiles (work items of tiles without any sample) whose cells the k-support pruned inverse grid FFT
// reads: the z range of the tile meets its tile column's wrapped kz hull [0, h) ∪ [nf2 − t, nf2) of the engine's
// support union (`hull[c] = (h, t)` per tile column c; (0, 0): column outside the support). One block per item; the
// other zero-count tiles are never read, so the knot-path scatter launches no blocks for them at all. Same cell walk
// as segnufft.cu seg_zero_split.
extern "C" __global__ void __launch_bounds__(256) seg_zero_empty(const int4* __restrict__ items, const int2* __restrict__ hull, cx* __restrict__ grid, Dims d) {
  const int4 it = items[blockIdx.x];
  const int nt0 = (d.nf0 + TT - 1) / TT, nt1 = (d.nf1 + TT - 1) / TT;
  const int c = it.x % (nt0 * nt1);
  const int2 hz = hull[c];
  if (hz.x == 0 && hz.y == 0) return;
  const int x0 = (c % nt0) * TT, y0 = (c / nt0) * TT, z0 = (it.x / (nt0 * nt1)) * TT;
  const int z1 = min(z0 + TT, d.nf2);
  if (!(z0 < hz.x || z1 > d.nf2 - hz.y)) return;   // block-uniform: the FFT never reads this tile
  const int run = min(TT, d.nf0 - x0) * NCH;
  for (int q = threadIdx.x; q < TT * TT * run; q += blockDim.x) {
    const int r = q / run, k = q % run;
    const int gy = y0 + r % TT, gz = z0 + r / TT;
    if (gy < d.nf1 && gz < d.nf2) grid[(((size_t)gz * d.nf1 + gy) * d.nf0 + x0) * NCH + k] = cx{0.f, 0.f};
  }
}

// Per-tile entry totals of a set's work items (the parts of a split tile summed): tcount[tile] += count (tcount zeroed
// by the host). The kl adjoint reads it in the first part of every split tile.
extern "C" __global__ void __launch_bounds__(256) seg_tile_count(const int4* __restrict__ items, int n, int* __restrict__ tcount) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int4 it = items[i];
    if (it.z > 0) atomicAdd(&tcount[it.x], it.z);
  }
}
