// llama_memory_seq_splice(): tokens cached at one place and moved must give the next token the
// logits it gets when the same tokens are decoded at the destination positions directly.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static bool decode(llama_context * ctx, const std::vector<llama_token> & tokens, size_t i0, size_t i1, llama_pos pos0, bool output_last) {
    common_batch batch(ctx);
    for (size_t i = i0; i < i1; i++) {
        batch.add(tokens[i], pos0 + (llama_pos) (i - i0), 0, output_last && i + 1 == i1);
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(logits, logits + n_vocab);
}

static int argmax(const std::vector<float> & v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx    = 2048;
    params.n_batch  = 512;
    params.prompt   = "The quick brown fox jumps over the lazy dog near the river bank. ";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::string text;
    for (int i = 0; i < 12; i++) {
        text += params.prompt;
    }
    const std::vector<llama_token> tokens = common_tokenize(ctx, text, true);
    const size_t n = tokens.size();
    const llama_pos shift = 301;

    llama_memory_t mem = llama_get_memory(ctx);

    if (!decode(ctx, tokens, 0, n - 1, 0, false) || !decode(ctx, tokens, n - 1, n, (llama_pos) n - 1, true)) {
        fprintf(stderr, "%s : failed to decode the reference run\n", __func__);
        return 1;
    }
    const std::vector<float> ref = last_logits(ctx, n_vocab);

    const auto max_diff = [&](const std::vector<float> & x, const std::vector<float> & y) {
        float d = 0.0f;
        for (int i = 0; i < n_vocab; i++) {
            d = std::max(d, std::fabs(x[i] - y[i]));
        }
        return d;
    };

    // the error a move must stay well below: the last token decoded `shift` positions away from unmoved tokens.
    // only memory that accepts a jump in positions can show it
    float misplaced_diff = -1.0f;
    llama_memory_clear(mem, true);
    if (decode(ctx, tokens, 0, n - 1, 0, false) && decode(ctx, tokens, n - 1, n, (llama_pos) n - 1 + shift, true)) {
        misplaced_diff = max_diff(last_logits(ctx, n_vocab), ref);
    }

    llama_memory_clear(mem, true);
    if (!decode(ctx, tokens, 0, n - 1, shift, false) || !decode(ctx, tokens, n - 1, n, (llama_pos) n - 1 + shift, true)) {
        fprintf(stderr, "%s : failed to decode the run at moved positions\n", __func__);
        return 1;
    }
    const std::vector<float> direct = last_logits(ctx, n_vocab);

    llama_memory_clear(mem, true);

    if (!decode(ctx, tokens, 0, n - 1, 0, false)) {
        fprintf(stderr, "%s : failed to decode the prefix\n", __func__);
        return 1;
    }

    const llama_memory_span overlap[] = {{0, 10, 0}, {5, 20, 0}};
    if (llama_memory_seq_splice(mem, 0, 0, overlap, 2)) {
        fprintf(stderr, "%s : FAILED - overlapping spans were accepted\n", __func__);
        return 1;
    }

    // a splice that keeps no prefix clears the recurrent state, so carry it across
    std::vector<uint8_t> rs(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    llama_state_seq_get_data_ext(ctx, rs.data(), rs.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    const llama_memory_span all[] = {{0, (llama_pos) n - 1, shift}};
    if (!llama_memory_seq_splice(mem, 0, 0, all, 1)) {
        fprintf(stderr, "%s : FAILED - splice was rejected\n", __func__);
        return 1;
    }

    // only hybrid memory has a recurrent state next to the moved cells
    const bool has_rs = llama_memory_seq_pos_max(mem, 0) == -1;
    if (has_rs) {
        llama_state_seq_set_data_ext(ctx, rs.data(), rs.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (!llama_memory_seq_rs_pos_set(mem, 0, (llama_pos) n - 2 + shift)) {
            fprintf(stderr, "%s : FAILED - could not move the recurrent state\n", __func__);
            return 1;
        }
    }

    // hybrid memory reports the recurrent position as its minimum
    if (llama_memory_seq_pos_min(mem, 0) < shift || llama_memory_seq_pos_max(mem, 0) != (llama_pos) n - 2 + shift) {
        fprintf(stderr, "%s : FAILED - positions after splice are [%d, %d]\n", __func__,
                llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    if (!decode(ctx, tokens, n - 1, n, (llama_pos) n - 1 + shift, true)) {
        fprintf(stderr, "%s : FAILED - could not decode after the splice\n", __func__);
        return 1;
    }
    const std::vector<float> moved = last_logits(ctx, n_vocab);

    fprintf(stderr, "%s : %zu tokens moved by %d: max logit difference %.5f to decoding at the destination, %.5f to the unmoved run (%.5f between those two, %.5f for a misplaced last token)\n", __func__,
            n - 1, shift, max_diff(moved, direct), max_diff(moved, ref), max_diff(direct, ref), misplaced_diff);

    if (argmax(direct) != argmax(moved) || max_diff(moved, direct) > 2.0f * max_diff(direct, ref) + 0.05f) {
        fprintf(stderr, "%s : FAILED - moved tokens do not behave like tokens decoded at the destination\n", __func__);
        return 1;
    }

    // remove a hole and refill it while later cells stay cached
    if (has_rs) {
        const llama_pos a = (llama_pos) n / 3;
        const llama_pos b = (llama_pos) n / 2;

        llama_memory_clear(mem, true);
        if (!decode(ctx, tokens, 0, n, 0, false)) {
            return 1;
        }

        const llama_memory_span tail[] = {{b, (llama_pos) n, 0}};
        if (!llama_memory_seq_splice(mem, 0, a, tail, 1)) {
            fprintf(stderr, "%s : FAILED - splice with a hole was rejected\n", __func__);
            return 1;
        }

        if (!llama_memory_seq_rs_pos_set(mem, 0, a - 1) || !decode(ctx, tokens, a, b, a, false)) {
            fprintf(stderr, "%s : FAILED - could not refill the hole\n", __func__);
            return 1;
        }

        if (!llama_memory_seq_rs_pos_set(mem, 0, (llama_pos) n - 1) || !decode(ctx, tokens, n - 1, n, (llama_pos) n, true)) {
            fprintf(stderr, "%s : FAILED - could not decode after refilling the hole\n", __func__);
            return 1;
        }

        fprintf(stderr, "%s : refilled a hole of %d tokens under %zu cached later tokens\n", __func__, b - a, n - b);
    }

    fprintf(stderr, "%s : SUCCESS\n", __func__);

    return 0;
}
