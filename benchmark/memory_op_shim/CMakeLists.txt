cmake_minimum_required(VERSION 3.19)
project(memory_test_ops LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

foreach(library MEMORY_ALLOC_LIBRARY MEMORY_WRITE_LIBRARY)
  if(NOT EXISTS "${${library}}")
    message(FATAL_ERROR "${library} must name a built memory target")
  endif()
endforeach()

add_library(memory_test_ops SHARED memory_op_shim.cpp)
target_compile_features(memory_test_ops PRIVATE cxx_std_20)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(memory_test_ops PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(
  memory_test_ops PRIVATE ${TORCH_LIBRARIES}
  "${MEMORY_ALLOC_LIBRARY}" "${MEMORY_WRITE_LIBRARY}"
)
get_filename_component(MEMORY_ALLOC_LIBRARY_DIR "${MEMORY_ALLOC_LIBRARY}" DIRECTORY)
get_filename_component(MEMORY_WRITE_LIBRARY_DIR "${MEMORY_WRITE_LIBRARY}" DIRECTORY)
set_target_properties(memory_test_ops PROPERTIES BUILD_RPATH
  "${MEMORY_ALLOC_LIBRARY_DIR};${MEMORY_WRITE_LIBRARY_DIR}")
