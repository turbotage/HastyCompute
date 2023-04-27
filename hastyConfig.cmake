get_filename_component(hasty_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Define the cmake installation directory:
set(hasty_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Provide all the library targets:
include("${CMAKE_CURRENT_LIST_DIR}/hastyTargets.cmake")

# Include all the custom cmake scripts ...

# Define the include directories:
set(hasty_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(hasty_LIBRARY_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../lib"     )
set(hasty_LIBRARYS      -lHastyComputeLib)

set(SupportedComponents HastyComputeLib)

set(hasty_FOUND True)

# Check that all the components are found:
# And add the components to the Foo_LIBS parameter:
foreach(comp ${hasty_FIND_COMPONENTS})
  if (NOT ";${SupportedComponents};" MATCHES comp)
    set(hasty_FOUND False)
    set(hasty_NOT_FOUND_MESSAGE "Unsupported component: ${comp}")
  endif()
  set(hasty_LIBS "${hasty_LIBS} -l{comp}")
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/hasty${comp}Targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/hasty${comp}Targets.cmake")
  endif()
endforeach()