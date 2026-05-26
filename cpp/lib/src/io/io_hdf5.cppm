module;

#include <nlohmann/json.hpp>
#include <highfive/highfive.hpp>

export module hasty_io_mod_hdf5;

import std;
import hasty_util_mod;

namespace hasty {
namespace io {
namespace hdf5 {

// <======================= JSON ==============================>
export nlohmann::json attr_to_json(HighFive::Attribute& attr)
{
	using TC = HighFive::DataTypeClass;
	auto tc     = attr.getDataType().getClass();
	bool scalar = attr.getSpace().getDimensions().empty();
	try {
		if (tc == TC::String) {
			if (scalar) { std::string v; attr.read(v); return v; }
			std::vector<std::string> v; attr.read(v);
			return v.size() == 1 ? nlohmann::json(v[0]) : nlohmann::json(v);
		}
		if (tc == TC::Integer || tc == TC::Enum) {
			if (scalar) { int64_t v; attr.read(v); return v; }
			std::vector<int64_t> v; attr.read(v);
			return v.size() == 1 ? nlohmann::json(v[0]) : nlohmann::json(v);
		}
		if (tc == TC::Float) {
			if (scalar) { double v; attr.read(v); return v; }
			std::vector<double> v; attr.read(v);
			return v.size() == 1 ? nlohmann::json(v[0]) : nlohmann::json(v);
		}
	} catch (...) {}
	return "<unparseable>";
}

// Collect all attributes of a group or dataset as a JSON object.
export template<typename HObj>
nlohmann::json all_attrs_json(HObj& obj)
{
	nlohmann::json j = nlohmann::json::object();
	for (const auto& name : obj.listAttributeNames()) {
		try {
			auto attr = obj.getAttribute(name);
			j[name]   = attr_to_json(attr);
		} catch (...) {
			j[name] = "<error>";
		}
	}
	return j;
}

// Raw HDF5 parse: returns (json_tree, unknown_type_attr_values, non_gv_dataset_paths).
// GenericValue conversion lives in hasty_generic_value_io_mod (generic_value_io.cppm).
export auto parse_hdf5_to_json(const std::string& filename, bool ignore_nongv_entries = true)
    -> Tup<nlohmann::json, std::vector<std::string>>;

}
}
}
