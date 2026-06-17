#  __  __       _         _____ __  __   _   _
# |  \/  | __ _(_)_ __   | ____|  \/  | | | | |
# | |\/| |/ _` | | '_ \  |  _| | |\/| | | | | |
# | |  | | (_| | | | | | | |___| |  | | | |_| |
# |_|  |_|\__,_|_|_| |_| |_____|_|  |_|  \___/

# ─── RPATH ────────────────────────────────────────────────────────────────────
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib;/usr/local/lib")
endif()

if(WIN32)
  add_compile_options(/wd4005)
elseif(APPLE)
  add_compile_options(-Wno-deprecated-literal-operator)
endif()

# ─── mads-onnx-inspect ─────────────────────────────────────────────────────────────
add_executable(mads-onnx-inspect onnx_inspect.cpp)

target_include_directories(mads-onnx-inspect
  PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${MADS_PREFIX}/include"
    "${ORT_INCLUDE_DIR}"
)

if(WIN32)
  target_link_libraries(mads-onnx-inspect PRIVATE onnxengine cxxopts Ws2_32 bcrypt)
else()
  target_link_libraries(mads-onnx-inspect PRIVATE onnxengine cxxopts)
endif()

set_target_properties(mads-onnx-inspect PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# ─── Install (gated) ──────────────────────────────────────────────────────────
if(MADS_INSTALL_AGENT)
  install(TARGETS mads-onnx-inspect RUNTIME DESTINATION bin)
endif()

# ─── onnx-bench (dev-only EP benchmark; not installed) ─────────────────────────
option(ONNX_BUILD_BENCH "Build the onnx-bench execution-provider benchmark" OFF)
if(ONNX_BUILD_BENCH)
  add_executable(onnx-bench onnx_bench.cpp)
  target_include_directories(onnx-bench
    PRIVATE
      "${CMAKE_SOURCE_DIR}/src"
      "${MADS_PREFIX}/include"
      "${ORT_INCLUDE_DIR}"
  )
  if(WIN32)
    target_link_libraries(onnx-bench PRIVATE onnxengine cxxopts Ws2_32 bcrypt)
  else()
    target_link_libraries(onnx-bench PRIVATE onnxengine cxxopts)
  endif()
  set_target_properties(onnx-bench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )
endif()

# ─── mads-onnx-source ─────────────────────────────────────────────────────────
# Only built when OpenCV was found by src/source/CMakeLists.txt
if(OPENCV_FOUND_IN_SOURCE)
  add_executable(mads-onnx-source onnx_source.cpp)

  target_include_directories(mads-onnx-source
    PRIVATE
      "${CMAKE_SOURCE_DIR}/src"
      "${MADS_PREFIX}/include"
      "${ORT_INCLUDE_DIR}"
  )

  if(WIN32)
    target_link_libraries(mads-onnx-source
      PRIVATE onnxengine onnxsource MadsCore cxxopts Ws2_32 bcrypt)
  else()
    target_link_libraries(mads-onnx-source
      PRIVATE onnxengine onnxsource MadsCore cxxopts)
  endif()

  set_target_properties(mads-onnx-source PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  # Install (gated)
  if(MADS_INSTALL_AGENT)
    install(TARGETS mads-onnx-source RUNTIME DESTINATION bin)
  endif()
endif()

# ─── mads-onnx-filter ─────────────────────────────────────────────────────────
add_executable(mads-onnx-filter onnx_filter.cpp)

target_include_directories(mads-onnx-filter
  PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${MADS_PREFIX}/include"
    "${ORT_INCLUDE_DIR}"
)

if(WIN32)
  target_link_libraries(mads-onnx-filter
    PRIVATE onnxengine MadsCore cxxopts Ws2_32 bcrypt)
else()
  target_link_libraries(mads-onnx-filter
    PRIVATE onnxengine MadsCore cxxopts)
endif()

set_target_properties(mads-onnx-filter PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# ─── Install (gated) ──────────────────────────────────────────────────────────
if(MADS_INSTALL_AGENT)
  install(TARGETS mads-onnx-filter RUNTIME DESTINATION bin)
endif()
