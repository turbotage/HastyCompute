module;

#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>

export module hasty_viz_mod;

import std;
import hasty_threading_mod;

namespace hasty {
namespace viz {

export template<typename T>
struct DefaultLinePlotsOptions {
    std::vector<std::span<T>> lines;
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
        std::vector<double> y;
        y.reserve(ln.size());
        for (const auto &v : ln) y.push_back(static_cast<double>(v));

        std::vector<double> x(y.size());
        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i);

        auto scatter = plotlypp::Scatter()
                           .x(x)
                           .y(y);

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



}
}