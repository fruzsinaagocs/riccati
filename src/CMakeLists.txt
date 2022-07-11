#set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
#set(LIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/lib)
#set(HEADER_FILES_DIR ${CMAKE_SOURCE_DIR}/include)
#set(HEADER_FILES ${HEADER_FILES_DIR}/bphf.hpp)

if(PYBIND11_INSTALLED)
  # # Create the Pybind11 C++ library
  # pybind11_add_module(bphf bphf.cpp)
  # target_include_directories(bphf PRIVATE ${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/ext/pybind11 ${CMAKE_SOURCE_DIR}/src)
  # set_target_properties(bphf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR})

  # Create executable for the tests
  add_executable(pybind_demo pybind_demo.cpp)
  # target_include_directories(test_pybind PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(pybind_demo PRIVATE pybind11::embed)
  set_target_properties(pybind_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif()

