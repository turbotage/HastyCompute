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



export Tensor decompress_ui16_config(const Tensor& q, const Config& cfg) {
    double a = getd(cfg, "a", 0.0);
    double b = getd(cfg, "b", 1.0);
    double clampa = getd(cfg, "clampa", a);
    double clampb = getd(cfg, "clampb", b);
    double focus  = getd(cfg, "focus", 1.0);

    std::string left_mode  = gets(cfg, "left_mode",  "linear");
    std::string right_mode = gets(cfg, "right_mode", "linear");

    double left_gamma  = getd(cfg, "left_gamma",  1.0);
    double right_gamma = getd(cfg, "right_gamma", 1.0);
    double left_logc   = getd(cfg, "left_logc",   1.0);
    double right_logc  = getd(cfg, "right_logc",  1.0);

    double t0 = (1.0 - focus) * 0.5;
    double t1 = 1.0 - t0;

    // Dequantize u16 → [0,1] float
    auto y = q.to(TensorOptions().dtype(eScalarType::Float)).div(65535.0);

    auto left_mask  = y.lt(t0);
    auto right_mask = y.gt(t1);

    // MIDDLE
    auto u_mid  = y.sub(t0).div(t1 - t0);
    auto result = u_mid.mul(b - a).add(a);

    // LEFT
    if (t0 > 0.0 && a > clampa) {
        auto u_left_shaped = y.div(t0);
        Tensor u_left;
        if (left_mode == "gamma") {
            u_left = pow(u_left_shaped, left_gamma);
        } else if (left_mode == "log") {
            u_left = expm1(u_left_shaped.mul(std::log1p(left_logc))).div(left_logc);
        } else {
            u_left = u_left_shaped;
        }
        result = where(left_mask, u_left.mul(a - clampa).add(clampa), result);
    }

    // RIGHT
    if (t1 < 1.0 && clampb > b) {
        auto u_right_shaped = y.sub(t1).div(1.0 - t1);
        Tensor u_right;
        if (right_mode == "gamma") {
            u_right = pow(u_right_shaped, right_gamma);
        } else if (right_mode == "log") {
            u_right = expm1(u_right_shaped.mul(std::log1p(right_logc))).div(right_logc);
        } else {
            u_right = u_right_shaped;
        }
        result = where(right_mask, u_right.mul(clampb - b).add(b), result);
    }

    return result;
}



export std::pair<Tensor, Config> compress_ui16_default(const Tensor& x) {
    auto xf   = x.to(TensorOptions().dtype(eScalarType::Float));
    auto flat = xf.flatten();

    double xmin = flat.min().item<float>();
    double xmax = flat.max().item<float>();

    double q01 = xmin;
    double q99 = xmax;

    if (xmax > xmin) {
        q01 = flat.quantile(0.01).item<float>();
        q99 = flat.quantile(0.99).item<float>();
        if (q01 >= q99) { q01 = xmin; q99 = xmax; }
    } else {
        q99 = xmin + 1.0;
        xmax = q99;
    }

    Config cfg;
    cfg["a"]          = q01;
    cfg["b"]          = q99;
    cfg["clampa"]     = xmin;
    cfg["clampb"]     = xmax;
    cfg["focus"]      = 1.0;
    cfg["left_mode"]  = std::string("linear");
    cfg["right_mode"] = std::string("linear");

    return { compress_ui16_config(xf, cfg), cfg };
}



}
}