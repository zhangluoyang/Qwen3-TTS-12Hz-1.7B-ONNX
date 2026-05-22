cmake_minimum_required(VERSION 3.20)
project(qwen3_tts_onnx_runtime LANGUAGES CXX)

# C++ runtime 构建说明：
# - qwen_onnx_runtime 是核心静态库；
# - apps/* 是不同学习/调试入口；
# - ONNX Runtime 由 cmake/OnnxRuntime.cmake 定位；
# - FFTW3f 用于 C++ mel_spectrogram，与 Python speaker encoder 前端对齐。

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QWEN_ORT_USE_CUDA "Enable CUDAExecutionProvider in ONNX Runtime sessions" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime release root containing include/ and lib/")
set(ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to libonnxruntime.so. Overrides ONNXRUNTIME_ROOT lib lookup")
set(CMAKE_BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/onnxruntime/capi")

include(cmake/OnnxRuntime.cmake)
find_package(Threads REQUIRED)

set(FFTW3F_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fftw3-local)
set(FFTW3F_INCLUDE_DIR ${FFTW3F_ROOT}/include)
set(FFTW3F_LIBRARY ${FFTW3F_ROOT}/lib/libfftw3f.a)
if(NOT EXISTS ${FFTW3F_INCLUDE_DIR}/fftw3.h OR NOT EXISTS ${FFTW3F_LIBRARY})
  message(FATAL_ERROR "Local FFTW3 float build not found. Build it under cpp/third_party/fftw3-local first.")
endif()

add_library(fftw3f STATIC IMPORTED)
set_target_properties(fftw3f PROPERTIES
  IMPORTED_LOCATION ${FFTW3F_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${FFTW3F_INCLUDE_DIR}
)

add_library(qwen_onnx_runtime
  src/audio_frontend.cc
  src/bpe_tokenizer.cc
  src/generation_config.cc
  src/ort_session.cc
  src/sampling.cc
  src/tensor.cc
  src/wav_writer.cc
  src/model_config.cc
  src/voice_clone_runtime.cc
)

target_include_directories(qwen_onnx_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(qwen_onnx_runtime
  PUBLIC
    ${ONNXRUNTIME_LIBRARY}
    Threads::Threads
    fftw3f
    ${CMAKE_DL_LIBS}
    m
)

if(QWEN_ORT_USE_CUDA)
  target_compile_definitions(qwen_onnx_runtime PUBLIC QWEN_ORT_USE_CUDA=1)
endif()

add_executable(qwen_ort_smoke apps/qwen_ort_smoke.cc)
target_link_libraries(qwen_ort_smoke PRIVATE qwen_onnx_runtime)

add_executable(qwen_voice_clone_skeleton apps/qwen_voice_clone_skeleton.cc)
target_link_libraries(qwen_voice_clone_skeleton PRIVATE qwen_onnx_runtime)

add_executable(qwen_voice_clone apps/qwen_voice_clone.cc)
target_link_libraries(qwen_voice_clone PRIVATE qwen_onnx_runtime)

add_executable(qwen_voice_clone_chunk apps/qwen_voice_clone_chunk.cc)
target_link_libraries(qwen_voice_clone_chunk PRIVATE qwen_onnx_runtime)

add_executable(qwen_custom_voice apps/qwen_custom_voice.cc)
target_link_libraries(qwen_custom_voice PRIVATE qwen_onnx_runtime)

add_executable(qwen_custom_voice_chunk apps/qwen_custom_voice_chunk.cc)
target_link_libraries(qwen_custom_voice_chunk PRIVATE qwen_onnx_runtime)

add_executable(qwen_voice_design apps/qwen_voice_design.cc)
target_link_libraries(qwen_voice_design PRIVATE qwen_onnx_runtime)

add_executable(qwen_voice_design_chunk apps/qwen_voice_design_chunk.cc)
target_link_libraries(qwen_voice_design_chunk PRIVATE qwen_onnx_runtime)

add_executable(qwen_prefill_debug apps/qwen_prefill_debug.cc)
target_link_libraries(qwen_prefill_debug PRIVATE qwen_onnx_runtime)

add_executable(qwen_decode_debug apps/qwen_decode_debug.cc)
target_link_libraries(qwen_decode_debug PRIVATE qwen_onnx_runtime)

add_executable(qwen_tokenizer_decode_debug apps/qwen_tokenizer_decode_debug.cc)
target_link_libraries(qwen_tokenizer_decode_debug PRIVATE qwen_onnx_runtime)
