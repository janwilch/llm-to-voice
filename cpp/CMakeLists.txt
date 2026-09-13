cmake_minimum_required(VERSION 3.20)
project(llmvoice_cpp C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake Tools reads this to drive IntelliSense in cpptools (see
# C_Cpp.default.configurationProvider in ../.vscode/settings.json).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Git REQUIRED)
find_package(Threads REQUIRED)

# --- Standard dependencies ----------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11
    GIT_TAG v2.7.2
)

FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(CLI11)

# --- One ggml for the whole process -------------------------------------------
#
# llama.cpp and qwentts.cpp each ship their own ggml. Keeping both fails on
# every platform, just at a different stage each way: as CMake subprojects they
# collide on the "ggml" target; linked statically into one binary they collide
# on every ggml_* symbol; as shared libraries, Windows reuses whichever
# ggml.dll loaded first (the loader matches DLLs by name) and Linux lets one
# copy's exported symbols interpose the other's. Even kept fully private
# inside two separate DLLs, it would still mean two GPU contexts, when sharing
# one runtime between the LLM and the TTS is the point (../docs/native-port.md).
# So there is exactly one ggml, and both link it.
#
# Which one: llama.cpp's, plus the two patches qwentts.cpp's ggml fork carries
# on top of upstream (patches/ggml/). llama.cpp is where ggml development
# happens, so its copy is the newest; the fork is upstream + those two commits
# and nothing else, verified against upstream's history rather than by diff.

# Static, position-independent: everything folds into one library for Unity,
# so there is no separate ggml.dll for another native plugin to clash with, and
# PIC because those static archives will end up inside a shared library.
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ABI-CRITICAL. qwentts.cpp's tensor names exceed ggml's default of 64 and it
# builds with 128. GGML_MAX_NAME sizes the name buffer inside struct
# ggml_tensor, so every translation unit that includes ggml.h must agree on it
# — a mismatch changes the struct layout, links cleanly, and corrupts memory.
# Directory scope, set before any add_subdirectory, so ggml, libllama,
# qwen-core and our own code all inherit the same value.
add_compile_definitions(GGML_MAX_NAME=128)

# What llama.cpp sets up before adding its own ggml. We add ggml first, so
# llama.cpp never gets to — without these we would silently lose its CPU
# matmul kernels (LLAMAFILE) and CUDA graph capture, which qwentts.cpp's decode
# loop also relies on, and MSVC would reject ggml's large objects (/bigobj)
# and mangle the non-ASCII literals in both projects (/utf-8).
if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()
if(NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>")
endif()

# --- Vendored dependencies ----------------------------------------------------
#
# Pinned to exact commits, not moving branches. Bump a GIT_TAG only
# deliberately; ../docs/native-port.md lists what to re-check when you do.
#
# SOURCE_SUBDIR names a directory that does not exist, which makes
# FetchContent_MakeAvailable download and pin the source without calling
# add_subdirectory on it. The order of the add_subdirectory calls below is the
# whole fix, so it is done by hand.
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        982937a3337f7e97ef08fd5603f4157575ece7e1
  PATCH_COMMAND  ${CMAKE_COMMAND}
                   -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml
                   -DPATCH_ROOT=ggml
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-patches.cmake
  SOURCE_SUBDIR  _populate_only
)

FetchContent_Declare(
  qwentts_cpp
  GIT_REPOSITORY https://github.com/ServeurpersoCom/qwentts.cpp.git
  GIT_TAG        779c7cb13a71ab7e39d543a332a56612670ff565
  # Its ggml submodule is the fork we replaced; do not fetch it at all.
  GIT_SUBMODULES ""
  SOURCE_SUBDIR  _populate_only
)

FetchContent_MakeAvailable(llama_cpp qwentts_cpp)

# ggml first, so that it exists as a target before llama.cpp looks for one.
# llama.cpp only adds its own "if (NOT TARGET ggml ...)" and is written to
# accept one from a parent project. EXCLUDE_FROM_ALL: build only what our
# targets actually link.
add_subdirectory(${llama_cpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR} EXCLUDE_FROM_ALL)

# qwentts.cpp's own CMakeLists.txt is not used. It adds its ggml
# unconditionally, resolves paths through ${CMAKE_SOURCE_DIR} (which is this
# directory once it is a subproject), redirects every target's output
# directory globally, and always builds five tools plus an HTTP server. Its
# library is three source files, so it is defined here instead. When bumping
# its GIT_TAG, diff its qwen-core target and top-level compile definitions
# against this block.
set(QWENTTS_DIR ${qwentts_cpp_SOURCE_DIR})

# version.h, generated with qwentts.cpp's own script so qt_version() reports
# exactly what an upstream build of this commit would.
execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DSRC_DIR=${QWENTTS_DIR}
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/qwentts/version.h
            -P ${QWENTTS_DIR}/tools/version.cmake
)

add_library(qwen-core STATIC
    ${QWENTTS_DIR}/src/qwen.cpp
    ${QWENTTS_DIR}/src/pipeline-tts.cpp
    ${QWENTTS_DIR}/src/pipeline-codec.cpp
)
# Vendored code builds at the standard it was written and tested against.
set_target_properties(qwen-core PROPERTIES CXX_STANDARD 17)
# PUBLIC: consumers of a static archive must not see dllimport on Windows.
target_compile_definitions(qwen-core PUBLIC QWEN_STATIC)
target_include_directories(qwen-core
    PUBLIC  ${QWENTTS_DIR}/src
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/qwentts
)
target_link_libraries(qwen-core PUBLIC ggml Threads::Threads)

# --- Our targets --------------------------------------------------------------

add_executable(llmvoice_cpp src/main.cpp
  src/tts/Qwen3TtsBackend.cpp)
target_link_libraries(llmvoice_cpp PRIVATE CLI11::CLI11 qwen-core)

target_compile_definitions(
    llmvoice_cpp PRIVATE
    QWEN_DEFAULT_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models"
)

enable_testing()

# Links libllama and qwen-core into one binary and checks they share one ggml
# with one GGML_MAX_NAME. If this builds and passes, the collision is solved.
add_executable(deps_smoke tests/deps_smoke.cpp)
target_link_libraries(deps_smoke PRIVATE llama qwen-core)
add_test(NAME deps_smoke COMMAND deps_smoke)

message(STATUS "llama.cpp   vendored at ${llama_cpp_SOURCE_DIR}")
message(STATUS "qwentts.cpp vendored at ${qwentts_cpp_SOURCE_DIR}")
