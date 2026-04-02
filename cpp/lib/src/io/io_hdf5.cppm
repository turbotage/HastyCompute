module;

#include <nlohmann/json.hpp>

export module hasty_io_mod:hdf5;

import std;
import hasty_util_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace io {
namespace hdf5 {


export GenericValue read_generic_value(const std::string& filename, bool ignore_nongv_entries = true);

// Read a single named entry from /gv_root/<entry_name> without loading the
// rest of the file.
export GenericValue read_generic_value_entry(const std::string& filename,
									  const std::string& entry_name,
									  bool ignore_nongv_entries = true);


export void write_generic_value(const GenericValue& value, const std::string& filename);

// Write (or overwrite) a single named entry /gv_root/<entry_name>.
// Creates the file and/or the gv_root group if they don't exist yet.
// Every other entry in the file is left untouched.
export void write_generic_value_entry(const GenericValue& value,
							   const std::string& filename,
							   const std::string& entry_name);



export auto parse_hdf5_as_generic_value(const std::string& filename, bool ignore_nongv_entries = true)
	-> Tup<GenericValue, nlohmann::json, std::vector<std::string>>;







}
}
}