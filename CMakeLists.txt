cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra -pedantic -Wno-vla")

project(heat_solver)
set(HEADERS
    date.h
    CLI11.hpp)
add_executable(${PROJECT_NAME} "main.cpp" ${HEADERS})
