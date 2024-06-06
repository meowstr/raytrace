cmake_minimum_required( VERSION 3.28 )

project( raytrace )

set( COMMON_SOURCES
  # includes
  src/logging.hpp

  # sources
  src/logging.cpp
  src/main.cpp
)

# common libs
add_library( cglm INTERFACE )
target_include_directories( cglm INTERFACE libs/cglm/include )

find_package( OpenMP REQUIRED )

#
# windows build
#
if ( (NOT DEFINED EMSCRIPTEN) AND WIN32 )
  # pull prebuilt libraries
  add_library( sdl-main STATIC IMPORTED )
  set_target_properties(sdl-main PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/libs/sdl/lib/SDL2main.lib)
  add_library( sdl SHARED IMPORTED )
  set_target_properties(sdl PROPERTIES IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/libs/sdl/lib/SDL2.lib)
  set_target_properties(sdl PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/libs/sdl/lib/SDL2.dll)
  target_include_directories(sdl INTERFACE libs/sdl/include)
  add_executable( app ${COMMON_SOURCES}  )
  set_target_properties(app PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)
  target_link_libraries( app PRIVATE cglm sdl sdl-main OpenMP::OpenMP_CXX )
endif()

# common build flags
target_include_directories( app PRIVATE src )
target_compile_features( app PRIVATE cxx_std_20 )
target_compile_definitions( app PRIVATE "RELEASE=$<CONFIG:Release>" )
add_custom_command(
  TARGET app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:app> $<TARGET_RUNTIME_DLLS:app>
  COMMAND_EXPAND_LISTS
)

