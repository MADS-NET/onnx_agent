#   ___  _   _ _   ___  __     _                    _
#  / _ \| \ | | \ | \ \/ /    / \   __ _  ___ _ __ | |_
# | | | |  \| |  \| |\  /    / _ \ / _` |/ _ \ '_ \| __|
# | |_| | |\  | |\  |/  \   / ___ \ (_| |  __/ | | | |_
#  \___/|_| \_|_| \_/_/\_\ /_/   \_\__, |\___|_| |_|\__|
#                                   |___/
cmake_minimum_required(VERSION 3.28)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# ─── Version ──────────────────────────────────────────────────────────────────
# Read from last git tag matching v*.*.*, default to 0.1.0
if(NOT DEFINED VERSION)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*.*.*" --abbrev=0
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_TAG_RAW
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(GIT_TAG_RAW MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(VERSION "${CMAKE_MATCH_1}")
    else()
      set(VERSION "0.1.0")
    endif()
  else()
    set(VERSION "0.1.0")
  endif()
endif()

project(onnx_agent VERSION "${VERSION}" LANGUAGES C CXX)

# ─── Project-wide settings ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
  add_compile_options(-fPIC)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# ─── MADS prefix discovery ────────────────────────────────────────────────────
set(MADS_ROOT "" CACHE PATH "Path to the MADS installation prefix")
if(MADS_ROOT)
  set(MADS_PREFIX_RAW "${MADS_ROOT}")
elseif(DEFINED ENV{MADS_ROOT} AND NOT "$ENV{MADS_ROOT}" STREQUAL "")
  set(MADS_PREFIX_RAW "$ENV{MADS_ROOT}")
else()
  execute_process(
    COMMAND mads -p
    RESULT_VARIABLE MADS_PREFIX_RESULT
    OUTPUT_VARIABLE MADS_PREFIX_RAW
    ERROR_VARIABLE  MADS_PREFIX_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT MADS_PREFIX_RESULT STREQUAL "0" OR MADS_PREFIX_RAW STREQUAL "")
    message(FATAL_ERROR
      "Could not determine the MADS installation prefix. "
      "Set -DMADS_ROOT=<path> or make sure 'mads -p' works."
    )
  endif()
endif()
string(STRIP "${MADS_PREFIX_RAW}" MADS_PREFIX_RAW)
file(TO_CMAKE_PATH "${MADS_PREFIX_RAW}" MADS_PREFIX)
set(MADS_ROOT "${MADS_PREFIX}" CACHE PATH "Path to the MADS installation prefix" FORCE)

if(NOT EXISTS "${MADS_PREFIX}/include/agent.hpp")
  message(FATAL_ERROR
    "MADS headers not found at '${MADS_PREFIX}/include'. "
    "Set -DMADS_ROOT to a valid MADS installation prefix."
  )
endif()

include_directories("${MADS_PREFIX}/include")
link_directories("${MADS_PREFIX}/lib")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${MADS_PREFIX}" CACHE PATH "Install prefix" FORCE)
endif()

# ─── CMP0169 policy ───────────────────────────────────────────────────────────
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
  cmake_policy(SET CMP0169 OLD)
endif()

# ─── Options ──────────────────────────────────────────────────────────────────
option(MADS_INSTALL_AGENT "Install the ONNX agents" OFF)

message(STATUS "Project:              ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "MADS_PREFIX:          ${MADS_PREFIX}")
message(STATUS "CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "MADS_INSTALL_AGENT:   ${MADS_INSTALL_AGENT}")

# ─── Dependencies via FetchContent ────────────────────────────────────────────
include(FetchContent)

# cxxopts (header-only, EXCLUDE_FROM_ALL)
FetchContent_Declare(cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.1.1
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
)

# ─── ONNX Runtime prebuilt download ───────────────────────────────────────────
set(ORT_VERSION "1.26.0" CACHE STRING "ONNX Runtime version to download")

# Build the archive name and URL based on platform/arch
if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(ORT_ARCHIVE_NAME "onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
  else()
    # x86_64 – not in latest releases; fall back gracefully; will error if not found
    set(ORT_ARCHIVE_NAME "onnxruntime-osx-x86_64-${ORT_VERSION}.tgz")
  endif()
elseif(LINUX)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ORT_ARCHIVE_NAME "onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
  else()
    set(ORT_ARCHIVE_NAME "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
  endif()
elseif(WIN32)
  set(ORT_ARCHIVE_NAME "onnxruntime-win-x64-${ORT_VERSION}.zip")
else()
  message(FATAL_ERROR "Unsupported platform for ONNX Runtime prebuilt download")
endif()

set(ORT_BASE_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}")
set(ORT_URL "${ORT_BASE_URL}/${ORT_ARCHIVE_NAME}")

message(STATUS "ORT_URL:              ${ORT_URL}")

FetchContent_Declare(onnxruntime_prebuilt
  URL       "${ORT_URL}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Silently populate (no add_subdirectory – it's a prebuilt)
set(_old_skip ${CMAKE_SKIP_INSTALL_RULES})
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_Populate(cxxopts)
add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)
FetchContent_Populate(onnxruntime_prebuilt)
set(CMAKE_SKIP_INSTALL_RULES ${_old_skip})

set(ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
message(STATUS "ORT_ROOT:             ${ORT_ROOT}")

# Locate include dir and library within the extracted tree
find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ORT_ROOT}/include"
  NO_DEFAULT_PATH
  REQUIRED
)

if(WIN32)
  find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
elseif(APPLE)
  find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
else()
  find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

message(STATUS "ORT_INCLUDE_DIR:      ${ORT_INCLUDE_DIR}")
message(STATUS "ORT_LIB:              ${ORT_LIB}")

# Imported target for convenience
add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED GLOBAL)
set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
  IMPORTED_LOCATION "${ORT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)
if(WIN32 AND EXISTS "${ORT_DLL}")
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_IMPLIB "${ORT_LIB}"
    IMPORTED_LOCATION "${ORT_DLL}"
  )
endif()

# ─── RPATH ────────────────────────────────────────────────────────────────────
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
elseif(LINUX)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib;/usr/local/lib")
endif()

# ─── ORT shared-library install (needed for self-contained package) ───────────
# Installs the runtime shared library into lib/ so that the RPATH
# (@executable_path/../lib on macOS, $ORIGIN/../lib on Linux) resolves.
if(MADS_INSTALL_AGENT)
  if(WIN32)
    # On Windows the DLL goes next to the executables (bin/) because that is
    # where the loader looks on Windows.  The import library is not needed at
    # runtime.
    install(FILES "${ORT_DLL}" DESTINATION bin)
  elseif(APPLE)
    # The binary links against @rpath/libonnxruntime.1.dylib.
    # Install the versioned dylib AND the .1 symlink so the soname resolves.
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    install(FILES
      "${_ORT_LIB_DIR}/libonnxruntime.${ORT_VERSION}.dylib"
      DESTINATION lib
    )
    # CMake install(FILES) dereferences symlinks, so re-create the .1 symlink
    # via a post-install script instead.
    install(CODE "
      set(_dst \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib\")
      file(CREATE_LINK
        libonnxruntime.${ORT_VERSION}.dylib
        \"\${_dst}/libonnxruntime.1.dylib\"
        SYMBOLIC
      )
    ")
  else()
    # Linux: install the versioned .so and recreate the .1 soname symlink.
    # The actual file name pattern is libonnxruntime.so.${ORT_VERSION}.
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    install(FILES
      "${_ORT_LIB_DIR}/libonnxruntime.so.${ORT_VERSION}"
      DESTINATION lib
    )
    install(CODE "
      set(_dst \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib\")
      file(CREATE_LINK
        libonnxruntime.so.${ORT_VERSION}
        \"\${_dst}/libonnxruntime.so.1\"
        SYMBOLIC
      )
      file(CREATE_LINK
        libonnxruntime.so.${ORT_VERSION}
        \"\${_dst}/libonnxruntime.so\"
        SYMBOLIC
      )
    ")
  endif()
endif()

# ─── Subdirectories ───────────────────────────────────────────────────────────
add_subdirectory("${SRC_DIR}/onnx")
add_subdirectory("${SRC_DIR}/source")
add_subdirectory("${SRC_DIR}/main")

# ─── CPack ────────────────────────────────────────────────────────────────────
string(TOLOWER "${CMAKE_SYSTEM_NAME}" PACKAGE_OS_NAME)
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  list(LENGTH CMAKE_OSX_ARCHITECTURES PACKAGE_OSX_ARCHITECTURES_COUNT)
  if(PACKAGE_OSX_ARCHITECTURES_COUNT GREATER 1)
    set(PACKAGE_ARCH_NAME "universal")
  else()
    set(PACKAGE_ARCH_NAME "${CMAKE_OSX_ARCHITECTURES}")
  endif()
else()
  set(PACKAGE_ARCH_NAME "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(PACKAGE_ARCH_NAME STREQUAL "")
  set(PACKAGE_ARCH_NAME "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
if(PACKAGE_ARCH_NAME STREQUAL "")
  set(PACKAGE_ARCH_NAME "unknown")
endif()
string(TOLOWER "${PACKAGE_ARCH_NAME}" PACKAGE_ARCH_NAME)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "MADS")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_PACKAGE_FILE_NAME
  "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${PACKAGE_OS_NAME}-${PACKAGE_ARCH_NAME}")
include(CPack)
