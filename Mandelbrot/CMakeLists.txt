project (Mandelbrot VERSION 1.0.0)

# Enable full optimisation
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets;-lineinfo")
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG};-Xptxas -v")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3")

find_package(CUDA REQUIRED)

# https://gitlab.kitware.com/cmake/cmake/issues/16602
set (CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})

cuda_include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/Include")

file (GLOB HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Include/*.hpp")
file (GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cu")

source_group ("Include\\" FILES ${HEADER_FILES})
source_group ("Source\\" FILES ${SOURCE_FILES})

cuda_add_executable (${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

add_dependencies (${PROJECT_NAME} Docopt)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "/arch:SSE2")
    target_compile_options(${PROJECT_NAME} PRIVATE "/openmp")
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE "-msse2")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fopenmp")
endif ()
        
set_property (TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

target_include_directories (${PROJECT_NAME} PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>)

target_link_libraries (${PROJECT_NAME} PUBLIC docopt)

add_custom_command (TARGET ${PROJECT_NAME} 
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE_DIR:docopt>/$<TARGET_FILE_NAME:docopt>" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")

install (TARGETS ${PROJECT_NAME}
	EXPORT ${TARGET_NAME}
	RUNTIME DESTINATION ${BINARY_INSTALL_DIR} 
    COMPONENT Applications)
