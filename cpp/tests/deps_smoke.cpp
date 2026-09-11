// Proves llama.cpp and qwentts.cpp share one ggml, built one way.
//
// That this links at all is the first half: both libraries are static, so two
// ggml copies would be two definitions of every ggml_* symbol. The second half
// is GGML_MAX_NAME, which a duplicate copy can get wrong without any link
// error, so it is checked from both sides of the boundary.

#include <cstdio>
#include <cstring>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "qwen.h"

// Our side: this translation unit sees the value qwentts.cpp needs.
static_assert(GGML_MAX_NAME == 128, "GGML_MAX_NAME must be 128 everywhere; see CMakeLists.txt");

// ggml's side. ggml_set_name truncates to the buffer size *ggml itself* was
// compiled with, so a 100-character name survives the round trip only if the
// ggml library agrees with the static_assert above. Metadata only: no_alloc
// means no tensor data is allocated.
static bool ggml_agrees_on_max_name() {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 2;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    char name[101];
    std::memset(name, 'x', 100);
    name[100] = '\0';

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t, name);
    const bool ok = std::strlen(ggml_get_name(t)) == 100;

    ggml_free(ctx);
    return ok;
}

int main() {
    llama_backend_init();

    std::printf("llama.cpp   : %s\n", llama_print_system_info());
    std::printf("qwentts.cpp : %s\n", qt_version());
    std::printf("ggml        : %zu backend(s) registered\n", ggml_backend_reg_count());

    const bool ok = ggml_agrees_on_max_name();
    std::printf("GGML_MAX_NAME=%d, ggml round-trips a 100-char name: %s\n",
                GGML_MAX_NAME, ok ? "yes" : "NO");

    llama_backend_free();
    return ok ? 0 : 1;
}
