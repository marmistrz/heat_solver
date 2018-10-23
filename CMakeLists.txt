cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra -pedantic -Wno-vla")

project(heat_solver)
add_executable(${PROJECT_NAME} "main.cpp")
