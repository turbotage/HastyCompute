import os

_dev_mode_build = True
_dev_mode_install = False

#D:\GitHub\HastyCompute\out\build\msvc-release-cuda

def get_ffi_libfile(debug=False, opsys="Windows"):
    if opsys == "Windows":
        libfile = "HastyPyInterface.dll"
        buildtype = "msvc-debug-cuda" if debug else "msvc-release-cuda"
    elif opsys == "Linux":
        libfile = "HastyPyInterface.so"
        buildtype = "clang-debug-cuda" if debug else "clang-release-cuda"

    if _dev_mode_build or _dev_mode_install:
        hc_dir = os.getcwd()
        dynlib_dir = os.path.join(hc_dir[:hc_dir.find("HastyCompute")], "HastyCompute", "out")

        dynlib_dir = os.path.join(dynlib_dir, "build" if _dev_mode_build else "install", 
			buildtype, "Debug" if debug else "Release")

    return os.path.join(dynlib_dir, libfile)
