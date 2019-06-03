cmake_minimum_required(VERSION 3.0)
project(heat_solver)
include(CheckCXXCompilerFlag)

set(AMPI_FLAGS "-memory isomalloc -module CommonLBs")
set(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
CHECK_CXX_COMPILER_FLAG("-memory isomalloc" AMPI)
if (AMPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AMPI_FLAGS}")
endif()

set(HEADERS
    date.h
    CLI11.hpp)
add_executable(${PROJECT_NAME} "main.cpp" ${HEADERS})
