module;

#include "tensor_spanning_view.hpp"

#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <plotlypp/traces/heatmap.hpp>

export module hasty_viz_mod;

import hasty_threading_mod;
import hasty_tensor_mod;
import hasty_util_mod;


namespace hasty {
namespace viz {

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
    Opt<Arr<Arr<std::string, N2>, N1>> titles = nullopt; // per-subplot titles
};

export template<std::size_t N1, std::size_t N2>
plotlypp::Figure default_heatmap(const DefaultHeatmapOptions<N1, N2>& opts)
{

    auto gridLayout = plotlypp::Layout().grid(
        plotlypp::Layout::Grid().rows(N1).columns(N2).pattern(plotlypp::Layout::Grid::Pattern::Independent));

    plotlypp::Figure fig;

    for_sequence<N1>([&opts, &fig](auto i) {
        for_sequence<N2>([&opts, &fig, i](auto j) {
            const auto &z = opts.z[i][j];
            // create heatmap trace for z
            plotlypp::Heatmap hm;
            hm.z(opts.z[i][j]);
            // set title if provided
            if (opts.titles && !(*opts.titles)[i][j].empty()) {
                hm.name((*opts.titles)[i][j]);
            }
            // assign trace to the correct subplot axes (x/x2, y/y2, ...)
            // Plotly uses "x", "x2", "x3"... and similarly for y axes. Subplot
            // numbering is 1-based in row-major order.
            const int subplot_index = static_cast<int>(i) * static_cast<int>(N2) + static_cast<int>(j) + 1;
            std::string xaxis = (subplot_index == 1) ? "x" : ("x" + std::to_string(subplot_index));
            std::string yaxis = (subplot_index == 1) ? "y" : ("y" + std::to_string(subplot_index));
            hm.json["xaxis"] = xaxis;
            hm.json["yaxis"] = yaxis;

            hm.showscale(true);
            // add to figure
            fig.addTrace(std::move(hm));
        });
    });

    fig.setLayout(std::move(gridLayout));
    return fig;
}

export void test_tensor_viz()
{
    auto rand1 = hasty::rand({100,200}, hasty::TensorOptions());
    auto rand2 = hasty::rand({100,100}, hasty::TensorOptions());

    default_heatmap(DefaultHeatmapOptions<1, 2>{
        .z = {{rand1.spanning_view(), rand2.spanning_view()}}
    }).show();
}


}
}