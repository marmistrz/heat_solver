cmake_minimum_required(VERSION 3.0)
project(heat_solver)

set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADERS
    date.h
    CLI11.hpp)
add_executable(${PROJECT_NAME} "main.cpp" ${HEADERS})
