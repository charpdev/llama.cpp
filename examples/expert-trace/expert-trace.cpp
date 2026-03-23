#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct callback_data {
    std::vector<uint8_t> data;
    int decode_calls = 0;
};

static bool has_name(const ggml_tensor * t, const char * want) {
    return t && std::strncmp(t->name, want, std::strlen(want)) == 0;
}

static void print_i32_tensor(const ggml_tensor * t, uint8_t * data, int layer_index, int decode_call_index) {
    if (t->type != GGML_TYPE_I32) return;

    const int64_t n_expert_used = t->ne[0];
    const int64_t n_tokens      = t->ne[1];

    std::printf("expert_trace layer=%d decode_call=%d shape=[%lld,%lld]\n",
            layer_index, decode_call_index,
            (long long) n_expert_used, (long long) n_tokens);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        std::printf("  token=%lld experts=", (long long) tok);
        for (int64_t ex = 0; ex < n_expert_used; ++ex) {
            const size_t  offset    = tok * t->nb[1] + ex * t->nb[0];
            const int32_t expert_id = *(int32_t *)(data + offset);
            std::printf("%s%d", ex == 0 ? "[" : ",", expert_id);
        }
        std::printf("]\n");
    }
}

static bool ggml_trace_experts(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    // Match ffn_moe_topk (with optional layer suffix like -0, -1...)
    const bool is_topk = t && std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (!is_topk) return false;
    if (ask) return true;

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    uint8_t * tensor_data = nullptr;

    if (is_host) {
        tensor_data = (uint8_t *) t->data;
    } else {
        const auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
        tensor_data = cb_data->data.data();
    }

    int layer_index = -1;
    // tensor name format: "ffn_moe_topk-<layer>"
    if (const char * dash = std::strrchr(t->name, '-')) {
        layer_index = std::atoi(dash + 1);
    }

    print_i32_tensor(t, tensor_data, layer_index, cb_data->decode_calls);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t)tokens.size()))) {
        std::fprintf(stderr, "%s: failed to eval\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    print_build_info();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = ggml_trace_experts;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    auto init  = common_init_from_params(params);
    auto * ctx = init->context();
    auto * mdl = init->model();

    if (!mdl || !ctx) {
        std::fprintf(stderr, "%s: failed to init\n", __func__);
        return 1;
    }

    const bool ok = run(ctx, params);

    llama_perf_context_print(ctx);
    llama_backend_free();

    return ok ? 0 : 1;
}
