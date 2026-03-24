module;

#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/bar.hpp>
#include <plotlypp/traces/pie.hpp>
#include <plotlypp/traces/scatter.hpp>

export module hasty_viz_mod;

import std;
import hasty_threading_mod;


namespace hasty {
namespace viz {

export plotlypp::Figure example_plot() 
{
    std::array<float, 4> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::span x(x_data);

    auto scatter = plotlypp::Scatter()
                        .x(x)
                        .y(std::vector{10,15,13,17})
                        .mode({plotlypp::Scatter::Mode::Markers})
                        .marker(
                            plotlypp::Scatter::Marker()
                                .color("rgb(82,64,219)")
                                .size(12)
                        )
                        .name("Markers");

    auto lines = plotlypp::Scatter()
                        .x(std::vector{2, 3, 4, 5})
                        .y(std::vector{16, 5, 11, 9})
                        .mode({plotlypp::Scatter::Mode::Lines})
                        .name("Lines");

    auto scatter_and_lines = plotlypp::Scatter()
                                 .x(std::vector{1, 2, 3, 4})
                                 .y(std::vector{12, 9, 15, 12})
                                 .mode({plotlypp::Scatter::Mode::Lines, plotlypp::Scatter::Mode::Markers})
                                 .name("Lines & Markers");

    auto layout = plotlypp::Layout()
                      .title([](auto& t) { t.text("This Graph's Title"); })
                      .xaxis(plotlypp::Layout::Xaxis().title([](auto& t) { t.text("x-axis title"); }))
                      .yaxis(plotlypp::Layout::Yaxis().title(plotlypp::Layout::Yaxis::Title().text("y-axis title")));
    return plotlypp::Figure()
        .addTraces(std::vector<plotlypp::Trace>{std::move(scatter), std::move(lines), std::move(scatter_and_lines)})
        .setLayout(std::move(layout));
}

}
}