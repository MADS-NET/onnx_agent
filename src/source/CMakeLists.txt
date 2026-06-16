#  ___  _   _ _   ___  __  ____
# / _ \| \ | | \ | \ \/ / / ___|  ___  _   _ _ __ ___ ___
#| | | |  \| |  \| |\  /  \___ \ / _ \| | | | '__/ __/ _ \
#| |_| | |\  | |\  |/  \   ___) | (_) | |_| | | | (_|  __/
# \___/|_| \_|_| \_/_/\_\ |____/ \___/ \__,_|_|  \___\___|

# ─── onnxsource static library ────────────────────────────────────────────────
# Always-compiled pieces (no OpenCV dependency)
add_library(onnxsource STATIC
  Classify.cpp
)

target_include_directories(onnxsource
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."   # enables <source/Classify.hpp>
    "${MADS_PREFIX}/include"
)

target_link_libraries(onnxsource
  PUBLIC onnxengine
)

set_target_properties(onnxsource PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# ─── OpenCV-dependent pieces ──────────────────────────────────────────────────
# Request only the modules we actually use to avoid pulling in heavy optional
# modules (e.g. opencv_sfm which requires gflags).
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)

if(OpenCV_FOUND)
  message(STATUS "OpenCV found: ${OpenCV_VERSION} — building CameraSource")

  # Collect only the four required module targets
  set(_OCV_NEEDED_LIBS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
  )

  target_compile_definitions(onnxsource PUBLIC HAVE_OPENCV)
  target_sources(onnxsource PRIVATE CameraSource.cpp)
  target_include_directories(onnxsource PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(onnxsource PUBLIC ${_OCV_NEEDED_LIBS})

  # Export a convenience variable for the parent scope
  set(OPENCV_FOUND_IN_SOURCE TRUE PARENT_SCOPE)
  set(OpenCV_LIBS_FOR_SOURCE "${_OCV_NEEDED_LIBS}" PARENT_SCOPE)
else()
  message(STATUS "OpenCV NOT found — CameraSource will be excluded")
  set(OPENCV_FOUND_IN_SOURCE FALSE PARENT_SCOPE)
endif()
