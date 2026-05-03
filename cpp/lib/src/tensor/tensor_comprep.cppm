module;

#include <nlohmann/json.hpp>

export module hasty_tensor_mod:comprep;

import std;
import :tensor;
import :external;
import :external_math;

namespace hasty {
namespace comprep {

    
export using Config = std::unordered_map<std::string, std::variant<double, std::string>>;

export std::string config_to_string(const Config& cfg) {
    nlohmann::json j;
    for (const auto& [key, value] : cfg) {
        if (std::holds_alternative<double>(value)) {
            j[key] = std::get<double>(value);
        } else if (std::holds_alternative<std::string>(value)) {
            j[key] = std::get<std::string>(value);
        }
    }
    return j.dump();
}

export Config string_to_config(const std::string& s) {
    Config cfg;
    auto j = nlohmann::json::parse(s);
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.value().is_number()) {
            cfg[it.key()] = it.value().get<double>();
        } else if (it.value().is_string()) {
            cfg[it.key()] = it.value().get<std::string>();
        }
    }
    return cfg;
}






inline double getd(const Config& cfg, const std::string& key, double def) {
    auto it = cfg.find(key);
    if (it == cfg.end()) return def;
    return std::get<double>(it->second);
}

inline std::string gets(const Config& cfg, const std::string& key, const std::string& def) {
    auto it = cfg.find(key);
    if (it == cfg.end()) return def;
    return std::get<std::string>(it->second);
}

Tensor quantize_to_u16(const Tensor& y) {
    auto y_clamped = clamp(y, 0.0, 1.0);
    return y_clamped.mul(65535.0).round().to(eScalarType::Short);
}



export Tensor compress_ui16_config(const Tensor& x, const Config& cfg) {
    // ---- required ----
    double a = getd(cfg, "a", 0.0);
    double b = getd(cfg, "b", 1.0);
    
    // ---- optional ----
    double clampa = getd(cfg, "clampa", a);
    double clampb = getd(cfg, "clampb", b);
    double focus  = getd(cfg, "focus", 1.0);
    
    std::string left_mode  = gets(cfg, "left_mode",  "linear");
    std::string right_mode = gets(cfg, "right_mode", "linear");
    
    double left_gamma  = getd(cfg, "left_gamma",  1.0);
    double right_gamma = getd(cfg, "right_gamma", 1.0);
    
    double left_logc   = getd(cfg, "left_logc",   1.0);
    double right_logc  = getd(cfg, "right_logc",  1.0);
    
    // ---- sanity (optional but recommended) ----
    // assume you have some assert mechanism
    // assert(clampa <= a && a < b && b <= clampb);
    // assert(focus > 0.0 && focus <= 1.0);
    
    // ---- precompute ----
    double t0 = (1.0 - focus) * 0.5;
    double t1 = 1.0 - t0;
    
    // ---- clamp ----
    auto xc = clamp(x, clampa, clampb);
    
    // ---- masks ----
    auto left_mask   = xc.lt(a);
    auto right_mask  = xc.gt(b);
    auto middle_mask = left_mask.logical_not().logical_and(right_mask.logical_not());
    
    // ---- LEFT ----
    auto u_left = xc.sub(clampa).div(a - clampa); // [0,1]
    
    Tensor u_left_shaped;
    if (left_mode == "gamma") {
        u_left_shaped = pow(u_left, 1.0 / left_gamma);
    } else if (left_mode == "log") {
        u_left_shaped = log1p(u_left.mul(left_logc)).div(std::log1p(left_logc));
    } else {
        u_left_shaped = u_left; // linear
    }
    
    auto y_left = u_left_shaped.mul(t0);
    
    // ---- MIDDLE ----
    auto u_mid = xc.sub(a).div(b - a);
    auto y_mid = u_mid.mul(t1 - t0).add(t0);
    
    // ---- RIGHT ----
    auto u_right = xc.sub(b).div(clampb - b);
    
    Tensor u_right_shaped;
    if (right_mode == "gamma") {
        u_right_shaped = pow(u_right, 1.0 / right_gamma);
    } else if (right_mode == "log") {
        u_right_shaped = log1p(u_right.mul(right_logc)).div(std::log1p(right_logc));
    } else {
        u_right_shaped = u_right;
    }
    
    auto y_right = u_right_shaped.mul(1.0 - t1).add(t1);
    
    // ---- combine ----
    auto y = y_mid;
    y = where(left_mask,  y_left,  y);
    y = where(right_mask, y_right, y);
    
    // ---- quantize ----
    auto q = quantize_to_u16(y);
    
    return q;
}



}
}