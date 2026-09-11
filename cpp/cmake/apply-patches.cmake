# Apply every *.patch in PATCH_DIR, in file-name order, to the git checkout in
# the current working directory (FetchContent runs PATCH_COMMAND from the
# dependency's source dir).
#
#   cmake -DGIT_EXECUTABLE=<git> -DPATCH_DIR=<dir> -DPATCH_ROOT=<subdir> -P apply-patches.cmake
#
# PATCH_ROOT is the subdirectory the patch paths are relative to. The ggml
# patches were cut from the standalone ggml repo, so inside llama.cpp they
# apply under ggml/.
#
# Idempotent on purpose: FetchContent can rerun the patch step on a tree it
# already patched, and failing there would break every reconfigure. A patch
# that reverse-applies cleanly is already in, so it is skipped.
#
# Editing a patch in place is the one case this cannot handle — the old version
# is already applied, so the new one neither applies nor reverses. Delete
# bin/_deps/llama_cpp-* and reconfigure after changing one.

file(GLOB patches "${PATCH_DIR}/*.patch")
list(SORT patches)

foreach(patch IN LISTS patches)
    get_filename_component(name "${patch}" NAME)

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "--directory=${PATCH_ROOT}" "${patch}"
        RESULT_VARIABLE not_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT not_applied)
        message(STATUS "patch already applied: ${name}")
        continue()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply "--directory=${PATCH_ROOT}" "${patch}"
        RESULT_VARIABLE failed
    )
    if(failed)
        message(FATAL_ERROR
            "patch does not apply: ${name}\n"
            "The pinned GIT_TAG moved underneath it. Rebase the patch onto the new "
            "tag, or drop it if upstream has since merged the change.")
    endif()
    message(STATUS "patch applied: ${name}")
endforeach()
