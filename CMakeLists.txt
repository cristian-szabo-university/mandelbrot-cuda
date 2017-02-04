cmake_minimum_required (VERSION 3.2)

###############################################################################
#
#   MAIN
#
###############################################################################

project (Mandelbrot VERSION 1.0.0)

if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
	message(FATAL_ERROR "In-tree builds are not supported; please run cmake in a separate build directory.")
endif ()

list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/Modules")
list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set (CMAKE_DEBUG_POSTFIX "_Debug")

set (TARGET_NAME "${CMAKE_PROJECT_NAME}Targets")
set (CONFIG_FILE "${CMAKE_PROJECT_NAME}-Config.cmake")
set (VERSION_FILE "${CMAKE_PROJECT_NAME}-Config-Version.cmake")
set (TARGET_FILE "${CMAKE_PROJECT_NAME}-Targets.cmake")

set (INCLUDE_INSTALL_DIR "Include")
set (LIBRARY_INSTALL_DIR "Library")
set (BINARY_INSTALL_DIR "Binary")
set (CONFIG_INSTALL_DIR "CMake")

###############################################################################
#
#   EXTERNAL PROJECTS
#
###############################################################################

include (ExternalProject)

set_directory_properties (PROPERTIES EP_BASE "${CMAKE_BINARY_DIR}/Extern")

ExternalProject_Add (
    Docopt

    GIT_REPOSITORY "https://github.com/docopt/docopt.cpp.git"
    GIT_TAG "master"
    
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

ExternalProject_Get_Property (Docopt SOURCE_DIR)
ExternalProject_Get_Property (Docopt BINARY_DIR)
ExternalProject_Get_Property (Docopt INSTALL_DIR)

file(MAKE_DIRECTORY "${INSTALL_DIR}/include/docopt")

add_library (docopt SHARED IMPORTED)

set_property (TARGET docopt PROPERTY CXX_STANDARD 11)

set_target_properties (docopt PROPERTIES
	IMPORTED_LOCATION "${INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}docopt${CMAKE_SHARED_LIBRARY_SUFFIX}"
    IMPORTED_IMPLIB "${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}docopt${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_DIR}/include/docopt")

###############################################################################
#
#   INTERNAL PROJECTS
#
###############################################################################

add_subdirectory ("${CMAKE_SOURCE_DIR}/mandelbrot-cpu")
add_subdirectory ("${CMAKE_SOURCE_DIR}/mandelbrot-gpu")

###############################################################################
#
#   INSTALL
#
###############################################################################

include (CMakePackageConfigHelpers)

configure_package_config_file (
	"${CMAKE_SOURCE_DIR}/ProjectConfig.cmake.inc" ${CONFIG_FILE}
	INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
	PATH_VARS INCLUDE_INSTALL_DIR LIBRARY_INSTALL_DIR BINARY_INSTALL_DIR)

write_basic_package_version_file (
	${VERSION_FILE}
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion)

install (FILES "${CMAKE_BINARY_DIR}/${CONFIG_FILE}" 
	"${CMAKE_BINARY_DIR}/${VERSION_FILE}"
    "${CMAKE_BINARY_DIR}/CMakeFiles/Export/CMake/${TARGET_FILE}"
    "${CMAKE_BINARY_DIR}/CMakeFiles/Export/CMake/${CMAKE_PROJECT_NAME}-Targets-$<$<CONFIG:Debug>:debug>$<$<CONFIG:Release>:release>.cmake"
    DESTINATION ${CONFIG_INSTALL_DIR}
    COMPONENT Libraries)
    
install (FILES "$<TARGET_FILE_DIR:docopt>/$<TARGET_FILE_NAME:docopt>"
    DESTINATION ${BINARY_INSTALL_DIR}
    COMPONENT Applications)

install (EXPORT ${TARGET_NAME}
	FILE ${TARGET_FILE}
	NAMESPACE "${CMAKE_PROJECT_NAME}::"
	DESTINATION ${CONFIG_INSTALL_DIR})

###############################################################################
#
#   CPACK
#
###############################################################################
    
set (CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Project Description")
set (CPACK_PACKAGE_VENDOR "Author Name")
set (CPACK_PACKAGE_CONTACT ${CPACK_PACKAGE_VENDOR})
set (CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set (CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set (CPACK_COMPONENT_HEADERS_DEPENDS Binaries Libraries)
set (CPACK_COMPONENT_APPLICATIONS_GROUP Runtime)
set (CPACK_COMPONENT_BINARIES_GROUP Development)
set (CPACK_COMPONENT_LIBRARIES_GROUP Development)
set (CPACK_COMPONENT_HEADERS_GROUP Development)

set (CPACK_ALL_INSTALL_TYPES Full Application Developer)
set (CPACK_COMPONENTS_ALL Applications Binaries Libraries Headers)
set (CPACK_COMPONENT_BINARIES_INSTALL_TYPES Application Developer Full)
set (CPACK_COMPONENT_LIBRARIES_INSTALL_TYPES Developer Full)
set (CPACK_COMPONENT_HEADERS_INSTALL_TYPES Developer Full)
set (CPACK_COMPONENT_APPLICATIONS_INSTALL_TYPES Application Full)
 
set (CPACK_PACKAGE_DIRECTORY "${PROJECT_SOURCE_DIR}/Package")
set (CPACK_STRIP_FILES FALSE)

include (CPack)
