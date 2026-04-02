module;

#include <nlohmann/json.hpp>
#include <highfive/highfive.hpp>

module hasty_io_mod;


import hasty_util_mod;
import hasty_tensor_mod;
import hasty_threading_mod;

namespace hasty {
namespace io {
namespace hdf5 {

constexpr const char* GV_TYPE_ATTR  = "gv_type";
constexpr const char* GV_DTYPE_ATTR = "gv_dtype";
constexpr const char* GV_SHAPE_ATTR = "gv_shape"; // only present for exotic (non-native HDF5) dtypes



// <======================= JSON ==============================>
static nlohmann::json attr_to_json(HighFive::Attribute& attr)
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
template<typename HObj>
static nlohmann::json all_attrs_json(HObj& obj)
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


// <======================= HELPERS ===========================>

bool is_native_hdf5_type(eScalarType st)
{
	switch (st) {
	case eScalarType::Byte:
	case eScalarType::Char:
	case eScalarType::Short:
	case eScalarType::Int:
	case eScalarType::Long:
	case eScalarType::Float:
	case eScalarType::Double:
	case eScalarType::ComplexFloat:
	case eScalarType::ComplexDouble:
	case eScalarType::Bool:
		return true;
	default:
		return false;
	}
}

// Try to map an HDF5 DataType to one of our eScalarType values.
// Uses HighFive's AtomicType equality comparison (backed by H5Tequal), which
// correctly matches both HighFive-written and externally-written datasets.
// Unsigned 16/32/64 and exotic compound types return nullopt.
static std::optional<eScalarType> hdf5_infer_scalar_type(const HighFive::DataType& dt)
{
	if (dt == HighFive::AtomicType<uint8_t>())  return eScalarType::Byte;
	if (dt == HighFive::AtomicType<int8_t>())   return eScalarType::Char;
	if (dt == HighFive::AtomicType<int16_t>())  return eScalarType::Short;
	if (dt == HighFive::AtomicType<int32_t>())  return eScalarType::Int;
	if (dt == HighFive::AtomicType<int64_t>())  return eScalarType::Long;
	if (dt == HighFive::AtomicType<float>())    return eScalarType::Float;
	if (dt == HighFive::AtomicType<double>())   return eScalarType::Double;
	if (dt == HighFive::AtomicType<std::complex<float>>())  return eScalarType::ComplexFloat;
	if (dt == HighFive::AtomicType<std::complex<double>>()) return eScalarType::ComplexDouble;
	// Bool tensors are written as uint8_t (b8), so they are already matched by
	// AtomicType<uint8_t> above.  HighFive v3 has no AtomicType<bool> specialization.
	// f16 has no native C++ type; detect by class + precision.
	if (dt.getClass() == HighFive::DataTypeClass::Float && dt.getSize() * 8 == 16)
		return eScalarType::Half;
	return std::nullopt;
}

// <======================= READ ===========================>

GenericValue read_tensor(HighFive::DataSet& ds);
GenericValue read_group_gv(HighFive::Group grp, bool ignore);
GenericValue read_dataset_gv(HighFive::DataSet& ds, bool ignore);
GenericValue read_gv(HighFive::Group& loc, const std::string& name, bool ignore);

GenericValue read_tensor(HighFive::DataSet& ds)
{
	std::string dtype_str;
	ds.getAttribute(GV_DTYPE_ATTR).read(dtype_str);
	eScalarType dtype = string_to_scalar_type(dtype_str);

	std::vector<uint8_t> raw_bytes;
	std::vector<i64> shape;

	if (ds.hasAttribute(GV_SHAPE_ATTR)) {
		// Exotic dtype: 1-D uint8 blob, shape stored in attribute
		std::vector<int64_t> gv_shape;
		ds.getAttribute(GV_SHAPE_ATTR).read(gv_shape);
		shape.assign(gv_shape.begin(), gv_shape.end());
		ds.read(raw_bytes);
	}
	else {
		// Native dtype: N-D typed dataset, read into correctly typed temp then memcpy
		auto hdims = ds.getDimensions();
		shape.assign(hdims.begin(), hdims.end());

		std::size_t numel = 1;
		for (auto d : hdims) numel *= d;

		auto read_native = [&]<typename T>() {
			std::vector<T> tmp(numel);
			ds.read_raw(tmp.data());
			raw_bytes.resize(numel * sizeof(T));
			std::memcpy(raw_bytes.data(), tmp.data(), raw_bytes.size());
		};
		switch (dtype) {
		case eScalarType::Byte:          read_native.operator()<uint8_t>();                  break;
		case eScalarType::Char:          read_native.operator()<int8_t>();                   break;
		case eScalarType::Short:         read_native.operator()<int16_t>();                  break;
		case eScalarType::Int:           read_native.operator()<int32_t>();                  break;
		case eScalarType::Long:          read_native.operator()<int64_t>();                  break;
		case eScalarType::Float:         read_native.operator()<float>();                    break;
		case eScalarType::Double:        read_native.operator()<double>();                   break;
		case eScalarType::ComplexFloat:  read_native.operator()<std::complex<float>>();     break;
		case eScalarType::ComplexDouble: read_native.operator()<std::complex<double>>();    break;
		case eScalarType::Bool:          read_native.operator()<uint8_t>();                  break;
		default:
			throw std::runtime_error(
				"Unexpected exotic dtype '" + dtype_str + "' without gv_shape attr (bf16/f16/c16 not natively supported)");
		}
	}

	return GenericValue(
		Tensor::from_vector(
			std::move(raw_bytes),
			ArrayRef<i64>(shape),
			dtype,
			Device(eDeviceType::CPU, DeviceIndex(-1))));
}

GenericValue read_group_gv(HighFive::Group grp, bool ignore)
{
	if (!grp.hasAttribute(GV_TYPE_ATTR)) {
		if (ignore) return GenericValue{};
		throw std::runtime_error(
			"HDF5 group '" + grp.getPath() + "' has no '" +
			GV_TYPE_ATTR + "' attribute; not a GenericValue");
	}

	std::string gv_type;
	grp.getAttribute(GV_TYPE_ATTR).read(gv_type);

	if (gv_type == "none")
		return GenericValue{};

	if (gv_type == "dict") {
		std::unordered_map<std::string, GenericValue> dict;
		for (const auto& child : grp.listObjectNames()) {
			try {
				dict.emplace(child, read_gv(grp, child, ignore));
			} catch (const std::exception&) {
				if (!ignore) throw;
			}
		}
		return GenericValue(std::move(dict));
	}

	if (gv_type == "vector" || gv_type == "tuple") {
		auto names = grp.listObjectNames();
		std::sort(names.begin(), names.end(),
			[](const std::string& a, const std::string& b) {
				return std::stoul(a) < std::stoul(b);
			});
		std::vector<GenericValue> elems;
		elems.reserve(names.size());
		for (const auto& child : names)
			elems.push_back(read_gv(grp, child, ignore));
		if (gv_type == "tuple")
			return GenericValue::make_tuple(std::move(elems));
		return GenericValue(std::move(elems));
	}

	if (ignore) return GenericValue{};
	throw std::runtime_error(
		"Unknown gv_type '" + gv_type + "' on group '" + grp.getPath() + "'");
}

GenericValue read_dataset_gv(HighFive::DataSet& ds, bool ignore)
{
	if (!ds.hasAttribute(GV_TYPE_ATTR)) {
		if (ignore) return GenericValue{};
		throw std::runtime_error(
			"HDF5 dataset '" + ds.getPath() + "' has no '" +
			GV_TYPE_ATTR + "' attribute; not a GenericValue");
	}

	std::string gv_type;
	ds.getAttribute(GV_TYPE_ATTR).read(gv_type);

	if (gv_type == "tensor")
		return read_tensor(ds);

	if (gv_type == "string") {
		std::string str;
		ds.read(str);
		return GenericValue(std::move(str));
	}

	if (ignore) return GenericValue{};
	throw std::runtime_error(
		"Unknown gv_type '" + gv_type + "' on dataset '" + ds.getPath() + "'");
}

GenericValue read_gv(HighFive::Group& loc, const std::string& name, bool ignore)
{
	auto obj_type = loc.getObjectType(name);
	if (obj_type == HighFive::ObjectType::Group)
		return read_group_gv(loc.getGroup(name), ignore);
	if (obj_type == HighFive::ObjectType::Dataset) {
		auto ds = loc.getDataSet(name);
		return read_dataset_gv(ds, ignore);
	}
	if (ignore) return GenericValue{};
	throw std::runtime_error(
		"HDF5 object '" + name + "' is neither a group nor a dataset");
}

GenericValue read_generic_value(const std::string& filename, bool ignore_nongv_entries)
{
	if (!std::filesystem::exists(filename))
		throw std::runtime_error("HDF5 file not found: '" + filename + "'");
	HighFive::File file(filename, HighFive::File::ReadOnly);
	auto root = file.getGroup("/");
	return read_gv(root, "gv_root", ignore_nongv_entries);
}

GenericValue read_generic_value_entry(const std::string& filename,
									  const std::string& entry_name,
									  bool ignore_nongv_entries)
{
	if (!std::filesystem::exists(filename))
		throw std::runtime_error("HDF5 file not found: '" + filename + "'");
	HighFive::File file(filename, HighFive::File::ReadOnly);
	auto root = file.getGroup("/");

	if (!root.exist("gv_root")) {
		if (ignore_nongv_entries) return GenericValue{};
		throw std::runtime_error("HDF5 file '" + filename + "' has no 'gv_root' group");
	}

	auto gv_root = root.getGroup("gv_root");

	if (!gv_root.exist(entry_name)) {
		if (ignore_nongv_entries) return GenericValue{};
		throw std::runtime_error(
			"Entry '" + entry_name + "' not found in 'gv_root' of '" + filename + "'");
	}

	return read_gv(gv_root, entry_name, ignore_nongv_entries);
}



// <======================= WRITE ===========================>

void write_tensor(HighFive::Group& loc, const std::string& name, const Tensor& tensor)
{
	Tensor cpu_t = (tensor.device().type != eDeviceType::CPU)
		? tensor.cpu().contiguous()
		: tensor.contiguous();

	const auto& sz  = cpu_t.sizes();
	const void* raw = cpu_t.const_data_ptr();
	const auto  st  = cpu_t.scalar_type();
	const std::string dtype_str = scalar_type_to_string(st);

	std::vector<std::size_t> shape(sz.begin(), sz.end());

	HighFive::DataSet ds;

	if (is_native_hdf5_type(st)) {
		// Store as a properly-typed N-D dataset (readable by h5py, HDFView, etc)
		HighFive::DataSpace space(shape);
		auto make = [&]<typename T>() {
			auto d = loc.createDataSet<T>(name, space);
			d.write_raw(static_cast<const T*>(raw));
			return d;
		};
		switch (st) {
		case eScalarType::Byte:          ds = make.operator()<u8>();                       break;
		case eScalarType::Char:          ds = make.operator()<i8>();                       break;
		case eScalarType::Short:         ds = make.operator()<i16>();                      break;
		case eScalarType::Int:           ds = make.operator()<i32>();                      break;
		case eScalarType::Long:          ds = make.operator()<i64>();                      break;
		case eScalarType::Float:         ds = make.operator()<f32>();                      break;
		case eScalarType::Double:        ds = make.operator()<f64>();                      break;
		case eScalarType::ComplexFloat:  ds = make.operator()<std::complex<float>>();     break;
		case eScalarType::ComplexDouble: ds = make.operator()<std::complex<double>>();    break;
		case eScalarType::Bool:          ds = make.operator()<b8>();                       break;
		default: break;
		}
		ds.createAttribute(GV_TYPE_ATTR,  std::string("tensor"));
		ds.createAttribute(GV_DTYPE_ATTR, dtype_str);
	}
	else {
		// Exotic dtype: store as 1-D uint8 blob + explicit gv_shape attribute
		std::size_t nbytes = static_cast<std::size_t>(cpu_t.numel() * scalar_type_size(st));
		const u8* raw_u8 = static_cast<const u8*>(raw);
		std::vector<u8> blob(raw_u8, raw_u8 + nbytes);
		ds = loc.createDataSet<u8>(name, HighFive::DataSpace({nbytes}));
		ds.write(blob);
		ds.createAttribute(GV_TYPE_ATTR,  std::string("tensor"));
		ds.createAttribute(GV_DTYPE_ATTR, dtype_str);
		std::vector<i64> gv_shape(sz.begin(), sz.end());
		ds.createAttribute(GV_SHAPE_ATTR, gv_shape);
	}
}

void write_gv(HighFive::Group& loc, const std::string& name, const GenericValue& value)
{
	switch (value.type()) {
        case GenericValue::eType::NONE: {
            auto grp = loc.createGroup(name);
            grp.createAttribute(GV_TYPE_ATTR, std::string("none"));
            break;
        }

        case GenericValue::eType::TENSOR:
            write_tensor(loc, name, value.as_tensor());
            break;

        case GenericValue::eType::VECTOR: {
            auto grp = loc.createGroup(name);
            grp.createAttribute(GV_TYPE_ATTR, std::string("vector"));
            const auto& vec = value.as_vector();
            for (size_t i = 0; i < vec.size(); ++i)
                write_gv(grp, std::to_string(i), vec[i]);
            break;
        }

        case GenericValue::eType::DICT: {
            auto grp = loc.createGroup(name);
            grp.createAttribute(GV_TYPE_ATTR, std::string("dict"));
            for (const auto& [key, val] : value.as_dict())
                write_gv(grp, key, val);
            break;
        }

        case GenericValue::eType::TUPLE: {
            auto grp = loc.createGroup(name);
            grp.createAttribute(GV_TYPE_ATTR, std::string("tuple"));
            const auto& vec = value.as_vector();
            for (size_t i = 0; i < vec.size(); ++i)
                write_gv(grp, std::to_string(i), vec[i]);
            break;
        }

        case GenericValue::eType::STRING: {
            // Store as a native HDF5 scalar string dataset
            const std::string& str = value.as_string();
            HighFive::DataSpace space = HighFive::DataSpace::From(str);
            auto ds = loc.createDataSet<std::string>(name, space);
            ds.write(str);
            ds.createAttribute(GV_TYPE_ATTR, std::string("string"));
            break;
        }
	}
}

void write_generic_value(const GenericValue& value, const std::string& filename)
{
	HighFive::File file(filename, HighFive::File::Truncate);
	auto root = file.getGroup("/");
	write_gv(root, "gv_root", value);
}

void write_generic_value_entry(const GenericValue& value,
							   const std::string& filename,
							   const std::string& entry_name)
{
	// Use ReadWrite|Create so we either open the existing file or create a fresh one.
	HighFive::File file(filename,
		HighFive::File::ReadWrite | HighFive::File::Create);

	auto root = file.getGroup("/");

	// Ensure gv_root group exists.
	HighFive::Group gv_root = root.exist("gv_root")
		? root.getGroup("gv_root")
		: root.createGroup("gv_root");

	if (!gv_root.hasAttribute(GV_TYPE_ATTR))
		gv_root.createAttribute(GV_TYPE_ATTR, std::string("dict"));

	// Remove the old entry so we can cleanly overwrite it.
	if (gv_root.exist(entry_name))
		gv_root.unlink(entry_name);

	write_gv(gv_root, entry_name, value);
}

// <======================= PARSING ===============================>

static std::string hdf5_dtype_str(const HighFive::DataType& dt)
{
	using TC  = HighFive::DataTypeClass;
	auto tc   = dt.getClass();
	auto bits = dt.getSize() * 8; // getSize() returns bytes
	if (tc == TC::Float) {
		if (bits == 16) return "float16";
		if (bits == 32) return "float32";
		if (bits == 64) return "float64";
		return "float" + std::to_string(bits);
	}
	if (tc == TC::Integer) {
		// Use the raw HDF5 sign query so we don't need a HighFive sign API.
		bool is_signed = (H5Tget_sign(dt.getId()) == H5T_SGN_2);
		return (is_signed ? "int" : "uint") + std::to_string(bits);
	}
	if (tc == TC::String)   return "string";
	if (tc == TC::Compound) return "compound";
	if (tc == TC::Enum)     return "enum";
	if (tc == TC::Array)    return "array";
	if (tc == TC::VarLen)   return "varlen";
	if (tc == TC::BitField) return "bitfield";
	if (tc == TC::Opaque)   return "opaque";
	return "unknown";
}


// Forward declaration (parse_group and parse_gv_object are mutually recursive).
static std::pair<GenericValue, nlohmann::json>
parse_gv_object(HighFive::Group& parent, const std::string& name,
				bool ignore, std::vector<std::string>& unparseable);

// Parse a single dataset.  If the dataset carries gv_type/gv_dtype attributes
// the existing typed reader is used; otherwise type inference is applied.
static std::pair<GenericValue, nlohmann::json>
parse_dataset(HighFive::DataSet& ds, bool ignore, std::vector<std::string>& unparseable)
{
	auto hdims = ds.getDimensions();
	std::vector<i64> shape(hdims.begin(), hdims.end());
	auto dt = ds.getDataType();

	nlohmann::json j;
	j["name"]       = ds.getPath();
	j["type"]       = "dataset";
	j["shape"]      = shape;
	j["dtype"]      = hdf5_dtype_str(dt);
	j["attributes"] = all_attrs_json(ds);

	// If gv_type is present, delegate to the existing typed reader.
	if (ds.hasAttribute(GV_TYPE_ATTR)) {
		try {
			std::string gv_type;
			ds.getAttribute(GV_TYPE_ATTR).read(gv_type);
			if (gv_type == "tensor") return {read_tensor(ds), j};
			if (gv_type == "string") {
				std::string str; ds.read(str);
				return {GenericValue(std::move(str)), j};
			}
		} catch (const std::exception& e) {
			unparseable.push_back(ds.getPath() + " (" + e.what() + ")");
			return {GenericValue{}, j};
		}
	}

	// No gv_type: infer from the HDF5 type class.
	using TC = HighFive::DataTypeClass;
	auto tc = dt.getClass();

	// String datasets → GenericValue string.
	if (tc == TC::String) {
		try {
			std::string str; ds.read(str);
			return {GenericValue(std::move(str)), j};
		} catch (...) {}
		try {
			std::vector<std::string> strs; ds.read(strs);
			if (strs.size() == 1) return {GenericValue(strs[0]), j};
		} catch (...) {}
		if (!ignore) unparseable.push_back(ds.getPath() + " (string array not supported)");
		return {GenericValue{}, j};
	}

	// Numeric: try to map to a scalar type and read raw bytes.
	auto st_opt = hdf5_infer_scalar_type(dt);
	if (!st_opt) {
		if (!ignore)
			unparseable.push_back(ds.getPath() + " (no scalar type mapping for '" + hdf5_dtype_str(dt) + "')");
		return {GenericValue{}, j};
	}
	eScalarType st = *st_opt;

	std::size_t numel = 1;
	for (auto d : hdims) numel *= d;

	try {
		std::vector<uint8_t> raw_bytes;
		if (st == eScalarType::Half) {
			// f16: no native HighFive type — read as packed uint16 bytes.
			raw_bytes.resize(numel * 2);
			ds.read_raw(reinterpret_cast<uint16_t*>(raw_bytes.data()));
		} else {
			auto read_native = [&]<typename T>() {
				std::vector<T> tmp(numel);
				ds.read_raw(tmp.data());
				raw_bytes.resize(numel * sizeof(T));
				std::memcpy(raw_bytes.data(), tmp.data(), raw_bytes.size());
			};
			switch (st) {
			case eScalarType::Byte:          read_native.operator()<uint8_t>();                break;
			case eScalarType::Char:          read_native.operator()<int8_t>();                 break;
			case eScalarType::Short:         read_native.operator()<int16_t>();                break;
			case eScalarType::Int:           read_native.operator()<int32_t>();                break;
			case eScalarType::Long:          read_native.operator()<int64_t>();                break;
			case eScalarType::Float:         read_native.operator()<float>();                  break;
			case eScalarType::Double:        read_native.operator()<double>();                 break;
			case eScalarType::ComplexFloat:  read_native.operator()<std::complex<float>>();   break;
			case eScalarType::ComplexDouble: read_native.operator()<std::complex<double>>();  break;
			case eScalarType::Bool:          read_native.operator()<uint8_t>();                break;
			default: break;
			}
		}
		Tensor tensor = Tensor::from_vector(
			std::move(raw_bytes), shape, st,
			Device(eDeviceType::CPU, DeviceIndex(-1)));
		return {GenericValue(std::move(tensor)), j};
	} catch (const std::exception& e) {
		if (!ignore) unparseable.push_back(ds.getPath() + " (" + e.what() + ")");
		return {GenericValue{}, j};
	}
}

static std::pair<GenericValue, nlohmann::json>
parse_group(HighFive::Group& grp, bool ignore, std::vector<std::string>& unparseable)
{
	nlohmann::json j;
	j["name"]       = grp.getPath();
	j["type"]       = "group";
	j["attributes"] = all_attrs_json(grp);
	j["children"]   = nlohmann::json::array();

	std::optional<std::string> gv_type_opt;
	if (grp.hasAttribute(GV_TYPE_ATTR)) {
		std::string gt; grp.getAttribute(GV_TYPE_ATTR).read(gt);
		gv_type_opt = gt;
	}

	if (gv_type_opt && *gv_type_opt == "none")
		return {GenericValue{}, j};

	bool is_ordered = gv_type_opt &&
		(*gv_type_opt == "tuple" || *gv_type_opt == "vector");
	bool make_tuple = gv_type_opt && *gv_type_opt == "tuple";

	auto child_names = grp.listObjectNames();
	if (is_ordered) {
		std::sort(child_names.begin(), child_names.end(),
			[](const std::string& a, const std::string& b) {
				try { return std::stoul(a) < std::stoul(b); }
				catch (...) { return a < b; }
			});
	}

	std::unordered_map<std::string, GenericValue> dict;
	std::vector<GenericValue> elems;

	for (const auto& child : child_names) {
		auto [child_gv, child_json] = parse_gv_object(grp, child, ignore, unparseable);
		j["children"].push_back(std::move(child_json));
		if (is_ordered)
			elems.push_back(std::move(child_gv));
		else
			dict.emplace(child, std::move(child_gv));
	}

	if (make_tuple) return {GenericValue::make_tuple(std::move(elems)), j};
	if (is_ordered) return {GenericValue(std::move(elems)), j};
	return {GenericValue(std::move(dict)), j};
}

static std::pair<GenericValue, nlohmann::json>
parse_gv_object(HighFive::Group& parent, const std::string& name,
				bool ignore, std::vector<std::string>& unparseable)
{
	try {
		auto obj_type = parent.getObjectType(name);
		if (obj_type == HighFive::ObjectType::Group) {
			auto grp = parent.getGroup(name);
			return parse_group(grp, ignore, unparseable);
		}
		if (obj_type == HighFive::ObjectType::Dataset) {
			auto ds = parent.getDataSet(name);
			return parse_dataset(ds, ignore, unparseable);
		}
	} catch (const std::exception& e) {
		if (!ignore)
			unparseable.push_back(parent.getPath() + "/" + name + " (" + e.what() + ")");
	}
	nlohmann::json j;
	j["name"] = parent.getPath() + "/" + name;
	j["type"] = "unknown";
	return {GenericValue{}, j};
}

Tup<GenericValue, nlohmann::json, std::vector<std::string>>
parse_hdf5_gv_impl(const std::string& filename, bool ignore)
{
	if (!std::filesystem::exists(filename))
		throw std::runtime_error("HDF5 file not found: '" + filename + "'");
	HighFive::File file(filename, HighFive::File::ReadOnly);
	auto root = file.getGroup("/");
	std::vector<std::string> unparseable;
	auto [gv, json_tree] = parse_group(root, ignore, unparseable);
	return {std::move(gv), std::move(json_tree), std::move(unparseable)};
}

auto parse_hdf5_as_generic_value(const std::string& filename, bool ignore_nongv_entries)
	-> Tup<GenericValue, nlohmann::json, std::vector<std::string>>
{
	return parse_hdf5_gv_impl(filename, ignore_nongv_entries);
}



}
}
}