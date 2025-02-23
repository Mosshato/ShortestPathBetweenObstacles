cmake_minimum_required(VERSION 3.20)
project(untitled1)

set(CMAKE_CXX_STANDARD 20)

# Set MinGW paths explicitly
set(CMAKE_INCLUDE_PATH "C:/msys64/MSYS2/mingw64/include")
set(CMAKE_LIBRARY_PATH "C:/msys64/MSYS2/mingw64/lib")

# Find OpenGL and GLFW
find_package(OpenGL REQUIRED)

# Include directories
include_directories("C:/msys64/MSYS2/mingw64/include")
include_directories("${CMAKE_SOURCE_DIR}")  # Ensure the project root is included (for stb_image.h)
link_directories("C:/msys64/MSYS2/mingw64/lib")

# Add executable
add_executable(untitled1 main.cpp)

# Link OpenGL, GLFW, and add stb_image dependencies
target_link_libraries(untitled1 OpenGL::GL glfw3)
