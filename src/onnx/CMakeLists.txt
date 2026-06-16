#  ___  _   _ _   ___  __  _____             _
# / _ \| \ | | \ | \ \/ / | ____|_ __   __ _(_)_ __   ___
#| | | |  \| |  \| |\  /  |  _| | '_ \ / _` | | '_ \ / _ \
#| |_| | |\  | |\  |/  \  | |___| | | | (_| | | | | |  __/
# \___/|_| \_|_| \_/_/\_\ |_____|_| |_|\__, |_|_| |_|\___|
#                                       |___/

add_library(onnxengine STATIC
  OnnxModel.cpp
  TensorJson.cpp
)

target_include_directories(onnxengine
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MADS_PREFIX}/include"
  PRIVATE
    "${ORT_INCLUDE_DIR}"
)

target_link_libraries(onnxengine
  PUBLIC  OnnxRuntime::OnnxRuntime
)

set_target_properties(onnxengine PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
