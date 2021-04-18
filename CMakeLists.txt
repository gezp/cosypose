cmake_minimum_required(VERSION 3.5)
project(cosypose)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install python
ament_python_install_package(${PROJECT_NAME})

# Install python scripts
install(PROGRAMS scripts/test.py DESTINATION bin)

ament_package()
