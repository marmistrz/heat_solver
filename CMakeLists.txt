cmake_minimum_required(VERSION 2.8)
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_FLAGS "-lm -Ofast -Wall -Wextra -pedantic -Wno-vla")

project(heat_solver)
add_executable(${PROJECT_NAME} "main.c")
