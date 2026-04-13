cmake_minimum_required(VERSION 3.16)
project(RotaOptimaldsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ROTA_COPY_CASADI_RUNTIME "Copy CasADi runtime libraries next to executable" ON)
set(CASADI_ROOT "" CACHE PATH "Root directory of CasADi C++ distribution (contains include/, lib/, bin/)")

set(_casadi_include_hints)
set(_casadi_library_hints)
set(_casadi_runtime_hints)

if(CASADI_ROOT)
  list(APPEND _casadi_include_hints "${CASADI_ROOT}" "${CASADI_ROOT}/include")
  list(APPEND _casadi_library_hints "${CASADI_ROOT}" "${CASADI_ROOT}/lib" "${CASADI_ROOT}/lib64" "${CASADI_ROOT}/bin")
  list(APPEND _casadi_runtime_hints "${CASADI_ROOT}" "${CASADI_ROOT}/bin" "${CASADI_ROOT}/lib" "${CASADI_ROOT}/lib64")
endif()

if(WIN32)
  list(APPEND _casadi_include_hints "C:/casadi/include")
  list(APPEND _casadi_library_hints "C:/casadi/lib" "C:/casadi/bin")
  list(APPEND _casadi_runtime_hints "C:/casadi/bin" "C:/casadi/lib")
elseif(APPLE)
  list(APPEND _casadi_include_hints /usr/local/include /opt/homebrew/include)
  list(APPEND _casadi_library_hints /usr/local/lib /opt/homebrew/lib)
  list(APPEND _casadi_runtime_hints /usr/local/lib /opt/homebrew/lib)
else()
  list(APPEND _casadi_include_hints /usr/local/include /usr/include)
  list(APPEND _casadi_library_hints /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
  list(APPEND _casadi_runtime_hints /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
endif()

find_path(CASADI_INCLUDE_DIR NAMES casadi/casadi.hpp HINTS ${_casadi_include_hints} PATH_SUFFIXES include)
find_library(CASADI_LIBRARY NAMES casadi libcasadi HINTS ${_casadi_library_hints} PATH_SUFFIXES lib lib64 bin)

if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
  message(FATAL_ERROR "CasADi C++ library not found. Set CASADI_INCLUDE_DIR and CASADI_LIBRARY.")
endif()

message(STATUS "CASADI_INCLUDE_DIR=${CASADI_INCLUDE_DIR}")
message(STATUS "CASADI_LIBRARY=${CASADI_LIBRARY}")

add_executable(rota_optimal_ds
  src/main.cpp
  src/colreg_classifier.cpp
  src/colreg_scenarios.cpp
  src/rota_optimal_ds.cpp
  src/obstacle_avoidance.cpp
  src/scenario_parser.cpp
)

target_include_directories(rota_optimal_ds PRIVATE src ${CASADI_INCLUDE_DIR})
target_link_libraries(rota_optimal_ds PRIVATE ${CASADI_LIBRARY})

if(UNIX AND NOT APPLE)
  get_filename_component(_casadi_lib_dir "${CASADI_LIBRARY}" DIRECTORY)
  set_target_properties(rota_optimal_ds PROPERTIES
    BUILD_RPATH "${_casadi_lib_dir}"
    INSTALL_RPATH "${_casadi_lib_dir}"
  )
endif()

set(CASADI_RUNTIME_FILES)
if(ROTA_COPY_CASADI_RUNTIME)
  if(WIN32)
    find_path(CASADI_RUNTIME_DIR NAMES casadi.dll HINTS ${_casadi_runtime_hints} PATH_SUFFIXES bin lib)
    if(CASADI_RUNTIME_DIR)
      file(GLOB _all_runtime_dlls "${CASADI_RUNTIME_DIR}/*.dll")
      foreach(_dll IN LISTS _all_runtime_dlls)
        get_filename_component(_name "${_dll}" NAME)
        string(TOLOWER "${_name}" _name_l)
        if(_name_l MATCHES "casadi|ipopt|coin|mumps|openblas|blasfeo")
          list(APPEND CASADI_RUNTIME_FILES "${_dll}")
        endif()
      endforeach()
      if(CASADI_RUNTIME_FILES)
        add_custom_command(TARGET rota_optimal_ds POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CASADI_RUNTIME_FILES}
            $<TARGET_FILE_DIR:rota_optimal_ds>
          COMMENT "Copying CasADi runtime DLLs"
        )
      else()
        message(WARNING "No runtime DLLs matched in ${CASADI_RUNTIME_DIR}. You may need to set PATH manually.")
      endif()
    else()
      message(WARNING "CASADI_RUNTIME_DIR not found; executable may fail to load plugins on Windows.")
    endif()
  endif()
endif()

install(TARGETS rota_optimal_ds RUNTIME DESTINATION .)
install(FILES plot_receding.py README.md DESTINATION .)
install(DIRECTORY scenarios DESTINATION .)
if(WIN32 AND CASADI_RUNTIME_FILES)
  install(FILES ${CASADI_RUNTIME_FILES} DESTINATION .)
endif()
