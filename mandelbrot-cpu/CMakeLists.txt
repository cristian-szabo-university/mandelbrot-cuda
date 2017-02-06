project (MandelbrotCPU VERSION 1.0.0)

option (ENABLE_PARALLEL_MODE "Enable multithreading support." OFF)
option (ENABLE_SSE_INTRINSICS "Enable sse intrinsics operations." OFF)

file (GLOB LOCAL_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Include/*.hpp")
file (GLOB LOCAL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp")

file (GLOB EXTERN_HEADER_FILES "${CMAKE_CURRENT_BINARY_DIR}/Include/*.hpp")
file (GLOB EXTERN_SOURCE_FILES "${CMAKE_CURRENT_BINARY_DIR}/Source/*.cpp")

configure_file (
	${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.inc 
	${CMAKE_CURRENT_BINARY_DIR}/Include/Config.hpp @ONLY)

set (HEADER_FILES
	${LOCAL_HEADER_FILES}
    ${EXTERN_HEADER_FILES})

source_group ("Include\\" FILES ${HEADER_FILES})

set (SOURCE_FILES
	${LOCAL_SOURCE_FILES}
    ${EXTERN_SOURCE_FILES})

source_group ("Source\\" FILES ${SOURCE_FILES})

add_executable (${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

add_dependencies (${PROJECT_NAME} Docopt)

if (ENABLE_SSE_INTRINSICS)
    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE "/arch:SSE2")
    else ()
        target_compile_options(${PROJECT_NAME} PRIVATE "-msse2")
    endif ()
endif ()

if (ENABLE_PARALLEL_MODE)
    if (MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE "/openmp")
    else ()
        target_compile_options(${PROJECT_NAME} PRIVATE "-fopenmp")
    endif ()
endif ()

set_property (TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

target_include_directories (${PROJECT_NAME} PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/Include>)

target_link_libraries (${PROJECT_NAME} PUBLIC docopt)

add_custom_command (TARGET ${PROJECT_NAME} 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE_DIR:docopt>/$<TARGET_FILE_NAME:docopt>" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")

install (TARGETS ${PROJECT_NAME}
	EXPORT ${TARGET_NAME}
	RUNTIME DESTINATION ${BINARY_INSTALL_DIR} 
    COMPONENT Applications)
