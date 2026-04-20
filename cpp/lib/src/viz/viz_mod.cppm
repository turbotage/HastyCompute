module;

#include "tensor_spanning_view.hpp"

#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <plotlypp/traces/heatmap.hpp>

export module hasty_viz_mod;

import hasty_threading_mod;
import hasty_tensor_mod;
import hasty_util_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace viz {

export class VizCache {
public:
    
    enum class PlotType : u16 {
        Orthoslicer
    };

    std::tuple<PlotType, std::string, GenericValue> pop_back() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_cache.empty()) {
            throw std::runtime_error("VizCache is empty");
        }
        auto item = m_cache.back();
        m_cache.pop_back();
        return item;
    }

    void push_back(PlotType type, std::string name, GenericValue data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.emplace_back(type, std::move(name), std::move(data));
    }

private:
    std::mutex m_mutex;
    std::vector<std::tuple<PlotType, std::string, GenericValue>> m_cache;

};

export extern VizCache global_viz_cache;

struct OrthoslicerOptions {
    std::string volumename;
};

export void orthoslicer(Tensor volume, const OrthoslicerOptions& options)
{
    auto uuid = generate_uuid();
    global_viz_cache.push_back(VizCache::PlotType::Orthoslicer, options.volumename, GenericValue(std::move(volume)));
}

export template<typename T>
struct DefaultLinePlotsOptions {
    std::vector<TensorSpanningView> lines;
    std::string title{};
    std::string xaxis{};
    std::string yaxis{};
    std::vector<std::string> legends{};
    std::vector<std::string> colors{};
    bool markers = true;
    bool lines_on = true;
};

export template<typename T>
plotlypp::Figure default_line_plots(const DefaultLinePlotsOptions<T>& opts)
{
    std::vector<plotlypp::Scatter> traces;
    traces.reserve(opts.lines.size());

    static const std::vector<std::string> default_palette = {
        "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
        "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
        "rgb(188,189,34)", "rgb(23,190,207)"
    };

    size_t li = 0;
    for (const auto &ln : opts.lines) {
        if (ln.ndim > 1)
            throw std::runtime_error("Lines must be dimension 1");

        std::vector<double> x(ln.sizes[0]);
        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i);

        auto scatter = plotlypp::Scatter()
                           .x(x)
                           .y(ln);

        if (opts.lines_on && opts.markers) {
            scatter.mode({plotlypp::Scatter::Mode::Lines, plotlypp::Scatter::Mode::Markers});
        } else if (opts.lines_on) {
            scatter.mode({plotlypp::Scatter::Mode::Lines});
        } else if (opts.markers) {
            scatter.mode({plotlypp::Scatter::Mode::Markers});
        } else {
            scatter.mode(plotlypp::Scatter::ModeExtra::None);
        }

        // choose color: user-provided or from default palette
        std::string color;
        if (!opts.colors.empty() && li < opts.colors.size()) color = opts.colors[li];
        else color = default_palette[li % default_palette.size()];

        scatter.line(plotlypp::Scatter::Line().color(color));
        scatter.marker(plotlypp::Scatter::Marker().color(color));

        if (!opts.legends.empty() && li < opts.legends.size()) scatter.name(opts.legends[li]);

        traces.push_back(std::move(scatter));
        ++li;
    }

    auto layout = plotlypp::Layout();
    if (!opts.title.empty()) layout.title([&](auto &t){ t.text(opts.title); });
    if (!opts.xaxis.empty()) layout.xaxis(plotlypp::Layout::Xaxis().title([&](auto &t){ t.text(opts.xaxis); }));
    if (!opts.yaxis.empty()) layout.yaxis(plotlypp::Layout::Yaxis().title(plotlypp::Layout::Yaxis::Title().text(opts.yaxis)));

    return plotlypp::Figure().addTraces(std::move(traces)).setLayout(std::move(layout));
}

export template<std::size_t N1, std::size_t N2>
struct DefaultHeatmapOptions {
    Arr<Arr<TensorSpanningView, N2>, N1> z;
    Opt<Arr<Arr<std::string, N2>, N1>> titles = nullopt;
};

export template<std::size_t N1, std::size_t N2>
struct DefaultHeatmapSliderOptions {
    // Each view must be 3-D: [NZ, NY, NX].
    // All cells must share the same NZ (slider dimension).
    Arr<Arr<TensorSpanningView, N2>, N1> z;
    Opt<Arr<Arr<std::string, N2>, N1>> titles = nullopt;
    std::string slider_prefix = "z = ";
};

export template<std::size_t N1, std::size_t N2>
plotlypp::Figure default_heatmap(const DefaultHeatmapOptions<N1, N2>& opts)
{
    plotlypp::Figure fig;
    auto layout = plotlypp::Layout();

    std::vector<plotlypp::Layout::Annotation> annotations;

    const double col_width  = 1.0 / static_cast<double>(N2);
    const double row_height = 1.0 / static_cast<double>(N1);


    const double cb_frac = 0.12;   // reserved space in cell
    const double cb_gap  = 0.005;   // small gap between plot and colorbar
    const double cb_width = 0.015;  // thickness (fraction mode)
    const double ygap = 0.03;

    for_sequence<N1>([&](auto i) {
        for_sequence<N2>([&](auto j) {

            const int idx =
                static_cast<int>(i) * static_cast<int>(N2)
              + static_cast<int>(j) + 1;

            std::string xref = (idx == 1) ? "x" : ("x" + std::to_string(idx));
            std::string yref = (idx == 1) ? "y" : ("y" + std::to_string(idx));

            // ---- Compute subplot "cell" ----
            double x0 = j * col_width;
            double x1 = (j + 1) * col_width;

            //double y1 = 1.0 - i * row_height;
            //double y0 = y1 - row_height;
            double total_ygap = (N1 - 1) * ygap;
            double row_height = (1.0 - total_ygap) / static_cast<double>(N1);

            double y1 = 1.0 - i * (row_height + ygap);
            double y0 = y1 - row_height;

            // shrink plot area to make space for colorbar INSIDE cell
            double plot_x1 = x1 - cb_frac * col_width;
            // ---- Colorbar inside subplot cell ----
            double cb_x = plot_x1 + cb_gap;
            double cb_y = (y0 + y1) * 0.5;

            // ---- Assign domains (THIS is the key part) ----
            layout.xaxis(idx, [&](auto& ax) {
                ax.domain({x0, plot_x1});
            });

            layout.yaxis(idx, [&](auto& ay) {
                ay.domain({y0, y1});
            });

            // ---- Heatmap ----
            plotlypp::Heatmap hm;
            hm.z(opts.z[i][j]);

            hm.json["xaxis"] = xref;
            hm.json["yaxis"] = yref;

            hm.showscale(true);
            hm.colorbar(
                plotlypp::Heatmap::Colorbar()
                    .x(cb_x)
                    .y(cb_y)
                    .len((y1 - y0) * 0.95)
                    .thicknessmode(plotlypp::Heatmap::Colorbar::Thicknessmode::Fraction)
                    .thickness(cb_width)
            );

            fig.addTrace(std::move(hm));

            // ---- Titles via annotations ----
            if (opts.titles && !(*opts.titles)[i][j].empty()) {

                double x_center = (x0 + plot_x1) * 0.5;
                double y_top    = y1 + 0.02;

                annotations.emplace_back(
                    plotlypp::Layout::Annotation()
                        .text((*opts.titles)[i][j])
                        .x(x_center)
                        .y(y_top)
                        .xref("paper")
                        .yref("paper")
                        .showarrow(false)
                );
            }

        });
    });

    if (!annotations.empty()) {
        layout.annotations(std::move(annotations));
    }

    fig.setLayout(std::move(layout));
    return fig;
}


// Returns a 2-D spanning view that is a z-slice at index `k` of a 3-D view.
// Strides and data pointers are adjusted; no copy is made.
TensorSpanningView impl_slice_z(const TensorSpanningView& v, std::int64_t k)
{
    if (v.ndim != 3)
        throw std::runtime_error("default_heatmap_slider: each z view must be 3-D [NZ,NY,NX]");

    std::size_t esz = 0;
    switch (v.simple_dtype) {
        case SimpleDType::F32: esz = 4; break;
        case SimpleDType::F64: esz = 8; break;
        case SimpleDType::I64: esz = 8; break;
        case SimpleDType::I32: esz = 4; break;
        case SimpleDType::I16: esz = 2; break;
        case SimpleDType::B8:  esz = 1; break;
        default: throw std::runtime_error("impl_slice_z: unsupported dtype");
    }

    TensorSpanningView s;
    s.simple_dtype = v.simple_dtype;
    s.ndim    = 2;
    s.sizes   = {v.sizes[1],   v.sizes[2]};
    s.strides = {v.strides[1], v.strides[2]};
    s.data    = reinterpret_cast<const std::uint8_t*>(v.data)
                + static_cast<std::size_t>(k * v.strides[0]) * esz;
    return s;
}

export template<std::size_t N1, std::size_t N2>
plotlypp::Figure default_heatmap_slider(const DefaultHeatmapSliderOptions<N1, N2>& opts)
{
    // Validate and determine NZ from the first non-null cell.
    std::int64_t NZ = -1;
    for_sequence<N1>([&](auto i) {
        for_sequence<N2>([&](auto j) {
            const auto& v = opts.z[i][j];
            if (v.ndim != 3)
                throw std::runtime_error("default_heatmap_slider: each z view must be 3-D [NZ,NY,NX]");
            if (NZ < 0) NZ = v.sizes[0];
            else if (v.sizes[0] != NZ)
                throw std::runtime_error("default_heatmap_slider: all cells must have the same NZ");
        });
    });
    if (NZ <= 0)
        throw std::runtime_error("default_heatmap_slider: NZ must be > 0");

    const int total_cells  = static_cast<int>(N1 * N2);
    const int total_traces = total_cells * static_cast<int>(NZ);

    plotlypp::Figure fig;
    auto layout = plotlypp::Layout();
    std::vector<plotlypp::Layout::Annotation> annotations;

    const double col_width = 1.0 / static_cast<double>(N2);
    const double cb_frac   = 0.12;
    const double cb_gap    = 0.005;
    const double cb_width  = 0.015;
    const double ygap      = 0.03;

    // Reserve vertical space at the bottom for the slider bar.
    const double slider_height = 0.10;
    const double plot_top      = 1.0;
    const double plot_bottom   = slider_height + 0.05;
    const double plot_span     = plot_top - plot_bottom;

    for_sequence<N1>([&](auto i) {
        for_sequence<N2>([&](auto j) {

            const int idx =
                static_cast<int>(i) * static_cast<int>(N2)
              + static_cast<int>(j) + 1;

            std::string xref = (idx == 1) ? "x" : ("x" + std::to_string(idx));
            std::string yref = (idx == 1) ? "y" : ("y" + std::to_string(idx));

            double x0 = j * col_width;
            double x1 = (j + 1) * col_width;

            double total_ygap = (N1 - 1) * ygap;
            double row_h = (plot_span - total_ygap) / static_cast<double>(N1);
            double y1 = plot_top  - i * (row_h + ygap);
            double y0 = y1 - row_h;

            double plot_x1 = x1 - cb_frac * col_width;
            double cb_x = plot_x1 + cb_gap;
            double cb_y = (y0 + y1) * 0.5;

            layout.xaxis(idx, [&](auto& ax) { ax.domain({x0, plot_x1}); });
            layout.yaxis(idx, [&](auto& ay) { ay.domain({y0, y1}); });

            // Emit one trace per z-slice; only slice 0 is initially visible.
            for (std::int64_t k = 0; k < NZ; ++k) {
                plotlypp::Heatmap hm;
                hm.z(impl_slice_z(opts.z[i][j], k));
                hm.json["xaxis"]   = xref;
                hm.json["yaxis"]   = yref;
                hm.json["visible"] = (k == 0);

                hm.showscale(true);
                hm.colorbar(
                    plotlypp::Heatmap::Colorbar()
                        .x(cb_x)
                        .y(cb_y)
                        .len((y1 - y0) * 0.95)
                        .thicknessmode(plotlypp::Heatmap::Colorbar::Thicknessmode::Fraction)
                        .thickness(cb_width)
                );
                fig.addTrace(std::move(hm));
            }

            if (opts.titles && !(*opts.titles)[i][j].empty()) {
                double x_center = (x0 + plot_x1) * 0.5;
                double y_top    = y1 + 0.02;
                annotations.emplace_back(
                    plotlypp::Layout::Annotation()
                        .text((*opts.titles)[i][j])
                        .x(x_center).y(y_top)
                        .xref("paper").yref("paper")
                        .showarrow(false)
                );
            }
        });
    });

    // Build slider: one step per z-slice.  Each step restylesall traces'
    // visibility so that only the slice-k traces for every cell are shown.
    std::vector<plotlypp::Layout::Slider::Step> steps;
    steps.reserve(static_cast<std::size_t>(NZ));

    for (std::int64_t k = 0; k < NZ; ++k) {
        plotlypp::Layout::Slider::Step step;
        step.label(std::to_string(k));
        step.method(plotlypp::Layout::Slider::Step::Method::Restyle);

        // args = [{"visible": [bool, ...]}, [0, 1, ..., total_traces-1]]
        // Trace layout: cell (i,j) owns traces at base = (i*N2+j)*NZ
        // so trace t is visible iff t % NZ == k.
        nlohmann::json vis_arr = nlohmann::json::array();
        for (int t = 0; t < total_traces; ++t)
            vis_arr.push_back(t % static_cast<int>(NZ) == static_cast<int>(k));

        nlohmann::json trace_idxs = nlohmann::json::array();
        for (int t = 0; t < total_traces; ++t)
            trace_idxs.push_back(t);

        step.json["args"] = nlohmann::json::array({
            nlohmann::json{{"visible", vis_arr}},
            trace_idxs
        });

        steps.push_back(std::move(step));
    }

    auto slider = plotlypp::Layout::Slider()
        .active(0.0)
        .steps(steps)
        .currentvalue([&](auto& cv) {
            cv.prefix(opts.slider_prefix);
            cv.visible(true);
        })
        .pad([](auto& p) { p.t(50.0); })
        .y(0.0)
        .yanchor(plotlypp::Layout::Slider::Yanchor::Top);

    layout.sliders({slider});

    if (!annotations.empty())
        layout.annotations(std::move(annotations));

    fig.setLayout(std::move(layout));
    return fig;
}



}
}