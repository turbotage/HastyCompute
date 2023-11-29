import os

_dev_mode_build = True
_dev_mode_install = False

#D:\GitHub\HastyCompute\out\build\msvc-release-cuda

def get_ffi_libfile():

    if _dev_mode_build:
        hc_dir = os.getcwd()
        dll_dir = os.path.join(hc_dir[:hc_dir.find("HastyCompute")], "out", "build", ...

    dll_path = ""
    return dll_path
