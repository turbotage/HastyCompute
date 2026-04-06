module;

export module hasty_fft_mod;

export import :nufft;
export import :toeplitz;


namespace hasty {
namespace fft {

export template<std::size_t N>
hasty::Tensor create_cartesian_coords(const std::array<hasty::i64, N>& im_size, TensorOptions options)
{
    using namespace hasty;
    constexpr float pi = 3.14159265358979f;

    std::array<hasty::i64, N> nmodes;
    std::reverse_copy(im_size.begin(), im_size.end(), nmodes.begin());

    i64 npts = 1;
    for (auto n : nmodes) npts *= n;

    std::vector<Tensor> coord_rows;
    coord_rows.reserve(N);

    for (std::size_t d = 0; d < N; ++d) {
        i64 n = nmodes[d];

        // inner = nmodes[0] * ... * nmodes[d-1]  → makes dim 0 cycle fastest
        i64 inner = 1;
        for (std::size_t k = 0; k < d; ++k) inner *= nmodes[k];

        // outer = nmodes[d+1] * ... * nmodes[N-1]
        i64 outer = 1;
        for (std::size_t k = d + 1; k < N; ++k) outer *= nmodes[k];

        // 1D grid for this dimension: 2*pi*k/n - pi, k = 0 .. n-1
        Tensor c1d = arange(n, options);
        c1d = c1d.mul(Scalar{2.0f * pi / static_cast<double>(n)}).add(Scalar{-pi});

        // Indices for one tile: each of the n values repeated `inner` times
        // [0,0,...(inner), 1,1,...(inner), ..., n-1,...(inner)]
        // Use float arange + divide + truncate-to-long (safe for non-negative values)
        Tensor fidx = arange(n * inner, options);
        Tensor idx  = fidx.div(Scalar{static_cast<double>(inner)}).to(eScalarType::Long);
        Tensor tile = c1d.index_select(0, idx);  // size = n * inner

        // Repeat the tile `outer` times along dim 0
        std::vector<Tensor> tiles;
        tiles.reserve(outer);
        for (i64 o = 0; o < outer; ++o) tiles.push_back(tile);

        coord_rows.push_back(cat(tiles, 0));  // size = npts
    }

    auto ret = hasty::stack(coord_rows, 0);  // [N, npts]
    coord_rows.clear();  // free memory
    ret = ret.to(options);
    return ret;
}

}
}