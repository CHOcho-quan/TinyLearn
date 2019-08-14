cmake_minimum_required(VERSION 3.10)

project(QSProject1_Cpp)

find_package(OpenCV)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/usr/local/include/eigen3/)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

add_executable(QSProject1_Cpp main.cpp featureGeneration.cpp includes.h featureGeneration.h models.cpp models.h dataset.cpp dataset.h)
target_link_libraries(QSProject1_Cpp ${OpenCV_LIBS})