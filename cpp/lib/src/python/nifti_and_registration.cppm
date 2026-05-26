module;

#include <nlohmann/json.hpp>

export module hasty_python_mod:nifti_and_registration;

import std;
import hasty_util_mod;
import hasty_generic_value_mod;
import hasty_server_mod;
import hasty_io_mod;
import hasty_io_mod_nifti;

namespace hasty {
namespace python {

export std::array<u8, 16> push_nifti_image(
    const io::nifti::NiftiImage& img,
    const std::string& name = "")
{
    auto uuid = server::global_generic_value_bank.push_value(
        GenericValue(img.data.clone()));

    nlohmann::json meta;
    meta["name"] = name;
    nlohmann::json pixdim_arr = nlohmann::json::array();
    for (auto v : img.header.pixdim) pixdim_arr.push_back(v);
    meta["pixdim"] = pixdim_arr;
    meta["qform_code"] = img.header.qform_code;
    meta["sform_code"] = img.header.sform_code;
    nlohmann::json qform_arr = nlohmann::json::array();
    nlohmann::json sform_arr = nlohmann::json::array();
    for (int i = 0; i < 4; ++i) {
        nlohmann::json row_q = nlohmann::json::array();
        nlohmann::json row_s = nlohmann::json::array();
        for (int j = 0; j < 4; ++j) {
            row_q.push_back(img.header.qform[i][j]);
            row_s.push_back(img.header.sform[i][j]);
        }
        qform_arr.push_back(row_q);
        sform_arr.push_back(row_s);
    }
    meta["qform"] = qform_arr;
    meta["sform"] = sform_arr;
    meta["scl_slope"] = img.header.scl_slope;
    meta["scl_inter"] = img.header.scl_inter;
    meta["description"] = img.header.description;

    std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
    server::global_generic_value_bank.write_metadata(key, meta.dump());

    return uuid;
}


}
}