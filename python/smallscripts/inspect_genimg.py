
import json
import h5py
import numpy as np


def convert_attr_value(val):
    """Convert HDF5 attribute values to JSON-serializable types."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore")
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.integer, np.floating)):
        return val.item()
    else:
        return val


def extract_attrs(obj):
    """Extract attributes from a group or dataset."""
    return {k: convert_attr_value(v) for k, v in obj.attrs.items()}


def visit_item(name, obj):
    """Recursively build structure for HDF5 items."""
    item_info = {
        "name": name,
        "type": "group" if isinstance(obj, h5py.Group) else "dataset",
        "attributes": extract_attrs(obj),
    }

    if isinstance(obj, h5py.Dataset):
        item_info.update({
            "shape": obj.shape,
            "dtype": str(obj.dtype),
        })

    if isinstance(obj, h5py.Group):
        item_info["children"] = []

    return item_info


def build_tree(h5file):
    """Build full hierarchical tree."""
    def recurse(obj, path="/"):
        node = {
            "name": path,
            "type": "group",
            "attributes": extract_attrs(obj),
            "children": []
        }

        for key in obj:
            item = obj[key]
            full_path = f"{path.rstrip('/')}/{key}"

            if isinstance(item, h5py.Dataset):
                node["children"].append({
                    "name": full_path,
                    "type": "dataset",
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                    "attributes": extract_attrs(item),
                })
            elif isinstance(item, h5py.Group):
                node["children"].append(recurse(item, full_path))

        return node

    return recurse(h5file)


def hdf5_to_json(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        tree = build_tree(f)
    return tree


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HDF5 structure to JSON")
    parser.add_argument("hdf5_file", type=str, help="Path to HDF5 file")
    args = parser.parse_args()

    result = hdf5_to_json(args.hdf5_file)

    # Pretty print JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))