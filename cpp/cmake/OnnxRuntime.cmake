set(_ORT_PY_LIB "${CMAKE_CURRENT_LIST_DIR}/../third_party/onnxruntime/lib/libonnxruntime.so")
set(_ORT_PY_DIR "/home/zhang/miniconda3/lib/python3.12/site-packages/onnxruntime/capi")
set(_ORT_VENDOR_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third_party/onnxruntime")
set(_ORT_VENDOR_INCLUDE "${_ORT_VENDOR_ROOT}/include")
set(_ORT_LOCAL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third_party/onnxruntime-local/onnxruntime-linux-x64-gpu-1.26.0")

if(NOT ONNXRUNTIME_ROOT AND EXISTS "${_ORT_LOCAL_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_ROOT "${_ORT_LOCAL_ROOT}" CACHE PATH "Path to an ONNX Runtime release root containing include/ and lib/" FORCE)
endif()

if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  if(NOT ONNXRUNTIME_LIB)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" NO_DEFAULT_PATH)
  endif()
endif()

if(ONNXRUNTIME_LIB)
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIB}")
endif()

if(NOT ONNXRUNTIME_LIBRARY AND EXISTS "${_ORT_PY_LIB}")
  set(ONNXRUNTIME_LIBRARY "${_ORT_PY_LIB}")
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR AND EXISTS "${_ORT_VENDOR_INCLUDE}/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_INCLUDE_DIR "${_ORT_VENDOR_INCLUDE}")
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(STATUS "ONNX Runtime C++ headers not found. Downloading headers into ${_ORT_VENDOR_ROOT}")
  file(MAKE_DIRECTORY "${_ORT_VENDOR_INCLUDE}")
  set(_ORT_BASE "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.26.0/include/onnxruntime/core/session")
  foreach(_h IN ITEMS
      onnxruntime_c_api.h
      onnxruntime_cxx_api.h
      onnxruntime_cxx_inline.h
      onnxruntime_float16.h
      onnxruntime_run_options_config_keys.h
      onnxruntime_session_options_config_keys.h
      onnxruntime_lite_custom_op.h
      onnxruntime_ep_c_api.h)
    file(DOWNLOAD "${_ORT_BASE}/${_h}" "${_ORT_VENDOR_INCLUDE}/${_h}" STATUS _status TLS_VERIFY ON)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      message(FATAL_ERROR "Failed to download ${_h}: ${_status}")
    endif()
  endforeach()
  set(ONNXRUNTIME_INCLUDE_DIR "${_ORT_VENDOR_INCLUDE}")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "libonnxruntime.so not found. Set ONNXRUNTIME_ROOT or ONNXRUNTIME_LIB.")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
