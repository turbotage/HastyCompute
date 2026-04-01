module;

#include <highfive/highfive.hpp>

export module hasty_io_mod:hdf5;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_threading_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace io {
namespace hdf5 {

constexpr const char* GV_TYPE_ATTR  = "gv_type";
constexpr const char* GV_DTYPE_ATTR = "gv_dtype";
constexpr const char* GV_SHAPE_ATTR = "gv_shape"; // only present for exotic (non-native HDF5) dtypes


void write_gv(HighFive::Group& loc, const std::string& name, const GenericValue& value);
GenericValue read_gv(HighFive::Group& loc, const std::string& name, bool ignore);
GenericValue read_group_gv(HighFive::Group grp, bool ignore);
GenericValue read_dataset_gv(HighFive::DataSet& ds, bool ignore);


GenericValue read_generic_value(const std::string& filename, bool ignore_nongv_entries)
{
	HighFive::File file(filename, HighFive::File::ReadOnly);
	auto root = file.getGroup("/");
	return read_gv(root, "gv_root", ignore_nongv_entries);
}

void write_generic_value(const GenericValue& value, const std::string& filename,
						 bool /*ignore_nongv_entries*/)
{
	HighFive::File file(filename, HighFive::File::Overwrite);
	auto root = file.getGroup("/");
	write_gv(root, "gv_root", value);
}

// Read a single named entry from /gv_root/<entry_name> without loading the
// rest of the file.
GenericValue read_generic_value_entry(const std::string& filename,
									  const std::string& entry_name,
									  bool ignore_nongv_entries)
{
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

// Write (or overwrite) a single named entry /gv_root/<entry_name>.
// Creates the file and/or the gv_root group if they don't exist yet.
// Every other entry in the file is left untouched.
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




// ── scalar type helpers ──────────────────────────────────────────────────────

// Returns true for types that map 1-to-1 onto a native HDF5 atomic type.
// f16, bf16, c32, c64 have no native HDF5 equivalent and are stored as raw bytes.
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
	case eScalarType::Bool:
		return true;
	default:
		return false;
	}
}
// ── tensor write helper ───────────────────────────────────────────────────────

void write_tensor(HighFive::Group& loc, const std::string& name, const Tensor& tensor)
{
	Tensor cpu_t = (tensor.device().type != eDeviceType::CPU)
		? tensor.cpu().contiguous()
		: tensor.contiguous();

	const auto& sz  = cpu_t.sizes();
	const void* raw = cpu_t.const_data_ptr();
	const auto  st  = cpu_t.scalar_type();
	const std::string dtype_str = scalar_type_to_string(st);

	std::vector<size_t> shape(sz.begin(), sz.end());

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
		case eScalarType::Byte:   ds = make.operator()<uint8_t>();  break;
		case eScalarType::Char:   ds = make.operator()<int8_t>();   break;
		case eScalarType::Short:  ds = make.operator()<int16_t>();  break;
		case eScalarType::Int:    ds = make.operator()<int32_t>();  break;
		case eScalarType::Long:   ds = make.operator()<int64_t>();  break;
		case eScalarType::Float:  ds = make.operator()<float>();    break;
		case eScalarType::Double: ds = make.operator()<double>();   break;
		case eScalarType::Bool:   ds = make.operator()<uint8_t>();  break; // bool == 1 byte
		default: break;
		}
		ds.createAttribute(GV_TYPE_ATTR,  std::string("tensor"));
		ds.createAttribute(GV_DTYPE_ATTR, dtype_str);
	}
	else {
		// Exotic dtype: store as 1-D uint8 blob + explicit gv_shape attribute
		size_t nbytes = static_cast<size_t>(cpu_t.numel() * scalar_type_size(st));
		const uint8_t* raw_u8 = static_cast<const uint8_t*>(raw);
		std::vector<uint8_t> blob(raw_u8, raw_u8 + nbytes);
		ds = loc.createDataSet<uint8_t>(name, blob);
		ds.createAttribute(GV_TYPE_ATTR,  std::string("tensor"));
		ds.createAttribute(GV_DTYPE_ATTR, dtype_str);
		std::vector<int64_t> gv_shape(sz.begin(), sz.end());
		ds.createAttribute(GV_SHAPE_ATTR, gv_shape);
	}
}

// ── tensor read helper ────────────────────────────────────────────────────────

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

		auto read_native = [&]<typename T>() {
			std::vector<T> tmp;
			ds.read(tmp);
			raw_bytes.resize(tmp.size() * sizeof(T));
			std::memcpy(raw_bytes.data(), tmp.data(), raw_bytes.size());
		};
		switch (dtype) {
		case eScalarType::Byte:   read_native.operator()<uint8_t>();  break;
		case eScalarType::Char:   read_native.operator()<int8_t>();   break;
		case eScalarType::Short:  read_native.operator()<int16_t>();  break;
		case eScalarType::Int:    read_native.operator()<int32_t>();  break;
		case eScalarType::Long:   read_native.operator()<int64_t>();  break;
		case eScalarType::Float:  read_native.operator()<float>();    break;
		case eScalarType::Double: read_native.operator()<double>();   break;
		case eScalarType::Bool:   read_native.operator()<uint8_t>();  break;
		default:
			throw std::runtime_error(
				"Unexpected exotic dtype '" + dtype_str + "' without gv_shape attr");
		}
	}

	return GenericValue(
		Tensor::from_vector(
			std::move(raw_bytes),
			ArrayRef<i64>(shape),
			dtype,
			Device(eDeviceType::CPU, DeviceIndex(-1))));
}

// ── recursive GV write ────────────────────────────────────────────────────────

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

// ── recursive GV read ─────────────────────────────────────────────────────────

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


}
}
}