# cmake/venv.cmake
# Creates a relocatable Python venv in the build dir, installs the hastycompute
# Python package into it, then installs the venv into the install prefix.
#
# The venv is created with --copies so it can be moved/installed to any path.
# C++ code invokes scripts as: ${VENV_PYTHON} script.py [args]
# where VENV_PYTHON is written into configure_file_settings.hpp.

set(HASTY_VENV_DIR "${CMAKE_BINARY_DIR}/.venv")
set(HASTY_PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python/lib")

# Stamp file — rebuild venv only when pyproject.toml changes.
set(_VENV_STAMP "${CMAKE_BINARY_DIR}/.venv_stamp")

add_custom_command(
    OUTPUT  "${_VENV_STAMP}"
    COMMAND "${Python3_EXECUTABLE}" -m venv --copies "${HASTY_VENV_DIR}"
    COMMAND "${HASTY_VENV_DIR}/bin/pip" install --quiet --upgrade pip
    COMMAND "${HASTY_VENV_DIR}/bin/pip" install --quiet
                setuptools grpcio grpcio-tools numpy torch antspyx debugpy
    COMMAND "${HASTY_VENV_DIR}/bin/pip" install --quiet -e "${HASTY_PYTHON_LIB_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_VENV_STAMP}"
    DEPENDS "${HASTY_PYTHON_LIB_DIR}/pyproject.toml"
    COMMENT "Creating Python venv and installing hastycompute package"
    VERBATIM
)

add_custom_target(hasty_venv ALL DEPENDS "${_VENV_STAMP}")
add_dependencies(HastyCompute hasty_venv)

# Install the venv into <prefix>/.venv (relocatable — uses copied binaries).
install(DIRECTORY "${HASTY_VENV_DIR}/"
    DESTINATION ".venv"
    USE_SOURCE_PERMISSIONS
)
