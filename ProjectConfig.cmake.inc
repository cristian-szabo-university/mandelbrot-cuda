@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

include ("${CMAKE_CURRENT_LIST_DIR}/@CMAKE_PROJECT_NAME@-Targets.cmake")

find_dependency(CUDA REQUIRED)
