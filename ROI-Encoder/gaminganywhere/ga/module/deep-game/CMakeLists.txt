cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

#add_executable(torchscript torchscript.cpp)
add_library(torchscript SHARED torchscript.cpp)
target_link_libraries(torchscript "${TORCH_LIBRARIES}")
set_property(TARGET torchscript PROPERTY CXX_STANDARD 14)