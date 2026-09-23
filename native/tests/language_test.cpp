#include "language_fixture.h"

#include "vn97/language.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using vn97_test::DenseMatVec;
using vn97_test::DenseRmsNorm;
using vn97_test::TinyLanguageFixture;

void ReferenceFullStep(
    const TinyLanguageFixture& fixture,
    const std::uint32_t* ids,
    std::size_t batch,
    std::vector<float>* state,
    std::vector<float>* logits,
    std::vector<float>* hidden) {
    constexpr std::size_t d = TinyLanguageFixture::kModel;
    constexpr std::size_t s = TinyLanguageFixture::kState;
    constexpr std::size_t v = TinyLanguageFixture::kVocab;

    for (std::size_t b = 0; b < batch; ++b) {
        float x[d] = {};
        for (std::size_t m = 0; m < d; ++m) {
            x[m] =
                fixture.embedding[
                    static_cast<std::size_t>(ids[b]) * d + m];
        }

        float norm[d] = {};
        float in_proj[2 * d] = {};
        float dt[d] = {};
        float b_vec[s] = {};
        float c_vec[s] = {};
        float selective[d] = {};
        float out_proj[d] = {};

        DenseRmsNorm(
            x,
            fixture.norm_weight.data(),
            d,
            fixture.model.rms_eps,
            norm);
        DenseMatVec(
            fixture.in_dense,
            2 * d,
            d,
            norm,
            nullptr,
            in_proj);
        DenseMatVec(
            fixture.dt_dense,
            d,
            d,
            in_proj,
            fixture.dt_bias.data(),
            dt);
        DenseMatVec(
            fixture.b_dense,
            s,
            d,
            in_proj,
            nullptr,
            b_vec);
        DenseMatVec(
            fixture.c_dense,
            s,
            d,
            in_proj,
            nullptr,
            c_vec);

        for (std::size_t m = 0; m < d; ++m) {
            const float bounded_dt =
                std::clamp(
                    std::log1p(std::exp(dt[m])),
                    fixture.model.dt_min,
                    fixture.model.dt_max);
            float sum = 0.0f;
            for (std::size_t n = 0; n < s; ++n) {
                const std::size_t state_index =
                    (b * d + m) * s + n;
                const float a =
                    fixture.a[m * s + n];
                const float z = a * bounded_dt;
                const float h =
                    std::exp(z) * (*state)[state_index] +
                    std::expm1(z) / a *
                        b_vec[n] *
                        in_proj[m];
                (*state)[state_index] = h;
                sum += h * c_vec[n];
            }
            const float gate = in_proj[d + m];
            selective[m] =
                sum *
                (gate /
                 (1.0f + std::exp(-gate)));
        }

        DenseMatVec(
            fixture.out_dense,
            d,
            d,
            selective,
            nullptr,
            out_proj);
        for (std::size_t m = 0; m < d; ++m) {
            x[m] += out_proj[m];
        }

        DenseRmsNorm(
            x,
            fixture.final_norm_weight.data(),
            d,
            fixture.model.rms_eps,
            norm);
        if (hidden != nullptr) {
            for (std::size_t m = 0; m < d; ++m) {
                (*hidden)[b * d + m] = norm[m];
            }
        }

        for (std::size_t token = 0;
             token < v;
             ++token) {
            float sum = 0.0f;
            for (std::size_t m = 0; m < d; ++m) {
                sum +=
                    norm[m] *
                    fixture.embedding[token * d + m];
            }
            (*logits)[b * v + token] = sum;
        }
    }
}

}  // namespace

int main() {
    TinyLanguageFixture fixture;
    assert(
        vn97::ValidateLanguageModel(fixture.model) ==
        vn97::LanguageStatus::kOk);

    std::size_t workspace_count = 0;
    assert(
        vn97::LanguageWorkspaceFloats(
            fixture.model,
            2,
            &workspace_count) ==
        vn97::LanguageStatus::kOk);

    std::vector<float> workspace(
        workspace_count,
        0.0f);
    std::vector<float> state(
        2 *
            TinyLanguageFixture::kModel *
            TinyLanguageFixture::kState,
        0.0f);
    std::vector<float> logits(
        2 * TinyLanguageFixture::kVocab,
        0.0f);
    const std::uint32_t ids[2] = {1, 3};

    assert(
        vn97::LanguageStepF32(
            fixture.model,
            ids,
            2,
            state.data(),
            logits.data(),
            logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOk);

    std::vector<float> reference_state(
        state.size(),
        0.0f);
    std::vector<float> reference_logits(
        logits.size(),
        0.0f);
    std::vector<float> reference_hidden(
        2 * TinyLanguageFixture::kModel,
        0.0f);
    ReferenceFullStep(
        fixture,
        ids,
        2,
        &reference_state,
        &reference_logits,
        &reference_hidden);

    for (std::size_t i = 0;
         i < logits.size();
         ++i) {
        assert(
            std::fabs(
                logits[i] -
                reference_logits[i]) <
            2e-5f);
    }
    for (std::size_t i = 0;
         i < state.size();
         ++i) {
        assert(
            std::fabs(
                state[i] -
                reference_state[i]) <
            2e-5f);
    }

    std::vector<float> hidden_state(
        2 *
            TinyLanguageFixture::kModel *
            TinyLanguageFixture::kState,
        0.0f);
    std::vector<float> hidden(
        2 * TinyLanguageFixture::kModel,
        0.0f);
    assert(
        vn97::LanguageStepHiddenF32(
            fixture.model,
            ids,
            2,
            hidden_state.data(),
            hidden.data(),
            hidden.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOk);
    for (std::size_t i = 0;
         i < hidden.size();
         ++i) {
        assert(
            std::fabs(
                hidden[i] -
                reference_hidden[i]) <
            2e-5f);
    }
    for (std::size_t i = 0;
         i < hidden_state.size();
         ++i) {
        assert(
            std::fabs(
                hidden_state[i] -
                reference_state[i]) <
            2e-5f);
    }

    std::uint32_t bad_token =
        TinyLanguageFixture::kVocab;
    assert(
        vn97::LanguageStepF32(
            fixture.model,
            &bad_token,
            1,
            state.data(),
            logits.data(),
            TinyLanguageFixture::kVocab,
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kInvalidToken);

    assert(
        vn97::LanguageStepF32(
            fixture.model,
            ids,
            2,
            state.data(),
            logits.data(),
            TinyLanguageFixture::kVocab,
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOutputTooSmall);

    std::vector<float> factors = {
        0.2f, 0.1f,
        -0.1f, 0.3f,
        0.4f, -0.2f,
        0.1f, 0.5f,
        -0.3f, 0.2f,
    };
    std::vector<float> projection = {
        0.7f, -0.2f, 0.1f,
        -0.1f, 0.6f, 0.3f,
    };

    fixture.model.embedding_kind =
        vn97::LanguageEmbeddingKind::kFactorizedF32;
    fixture.model.embedding = nullptr;
    fixture.model.embedding_rank =
        TinyLanguageFixture::kRank;
    fixture.model.token_factors =
        factors.data();
    fixture.model.projection =
        projection.data();

    assert(
        vn97::ValidateLanguageModel(fixture.model) ==
        vn97::LanguageStatus::kOk);
    assert(
        vn97::LanguageWorkspaceFloats(
            fixture.model,
            1,
            &workspace_count) ==
        vn97::LanguageStatus::kOk);

    workspace.assign(
        workspace_count,
        0.0f);
    state.assign(
        TinyLanguageFixture::kModel *
            TinyLanguageFixture::kState,
        0.0f);
    logits.assign(
        TinyLanguageFixture::kVocab,
        0.0f);

    const std::uint32_t factorized_token = 2;
    assert(
        vn97::LanguageStepF32(
            fixture.model,
            &factorized_token,
            1,
            state.data(),
            logits.data(),
            logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOk);

    const auto factorized_state = state;
    const auto factorized_logits = logits;

    std::vector<float> effective(
        TinyLanguageFixture::kVocab *
            TinyLanguageFixture::kModel,
        0.0f);
    for (std::size_t token_index = 0;
         token_index < TinyLanguageFixture::kVocab;
         ++token_index) {
        for (std::size_t m = 0;
             m < TinyLanguageFixture::kModel;
             ++m) {
            float sum = 0.0f;
            for (std::size_t r = 0;
                 r < TinyLanguageFixture::kRank;
                 ++r) {
                sum +=
                    factors[
                        token_index *
                            TinyLanguageFixture::kRank +
                        r] *
                    projection[
                        r *
                            TinyLanguageFixture::kModel +
                        m];
            }
            effective[
                token_index *
                    TinyLanguageFixture::kModel +
                m] = sum;
        }
    }

    auto full_equivalent = fixture.model;
    full_equivalent.embedding_kind =
        vn97::LanguageEmbeddingKind::kFullF32;
    full_equivalent.embedding_rank = 0;
    full_equivalent.embedding = effective.data();
    full_equivalent.token_factors = nullptr;
    full_equivalent.projection = nullptr;

    assert(
        vn97::ValidateLanguageModel(
            full_equivalent) ==
        vn97::LanguageStatus::kOk);

    std::vector<float> equivalent_state(
        TinyLanguageFixture::kModel *
            TinyLanguageFixture::kState,
        0.0f);
    std::vector<float> equivalent_logits(
        TinyLanguageFixture::kVocab,
        0.0f);
    assert(
        vn97::LanguageStepF32(
            full_equivalent,
            &factorized_token,
            1,
            equivalent_state.data(),
            equivalent_logits.data(),
            equivalent_logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOk);

    for (std::size_t i = 0;
         i < factorized_state.size();
         ++i) {
        assert(
            std::fabs(
                factorized_state[i] -
                equivalent_state[i]) <
            2e-5f);
    }
    for (std::size_t i = 0;
         i < factorized_logits.size();
         ++i) {
        assert(
            std::fabs(
                factorized_logits[i] -
                equivalent_logits[i]) <
            2e-5f);
    }

    std::vector<float> embedding_state(
        TinyLanguageFixture::kModel *
            TinyLanguageFixture::kState,
        0.0f);
    std::vector<float> embedding_logits(
        TinyLanguageFixture::kVocab,
        0.0f);
    const float* effective_embedding =
        effective.data() +
        static_cast<std::size_t>(factorized_token) *
            TinyLanguageFixture::kModel;

    assert(
        vn97::LanguageStepEmbeddingsF32(
            full_equivalent,
            effective_embedding,
            TinyLanguageFixture::kModel,
            1,
            embedding_state.data(),
            embedding_logits.data(),
            embedding_logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kOk);

    for (std::size_t i = 0;
         i < embedding_state.size();
         ++i) {
        assert(
            std::fabs(
                embedding_state[i] -
                equivalent_state[i]) <
            2e-5f);
    }
    for (std::size_t i = 0;
         i < embedding_logits.size();
         ++i) {
        assert(
            std::fabs(
                embedding_logits[i] -
                equivalent_logits[i]) <
            2e-5f);
    }

    assert(
        vn97::LanguageStepEmbeddingsF32(
            full_equivalent,
            effective_embedding,
            TinyLanguageFixture::kModel - 1,
            1,
            embedding_state.data(),
            embedding_logits.data(),
            embedding_logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kInvalidConfig);

    auto non_finite_embedding = std::vector<float>(
        effective_embedding,
        effective_embedding + TinyLanguageFixture::kModel);
    non_finite_embedding[1] =
        std::numeric_limits<float>::quiet_NaN();
    assert(
        vn97::LanguageStepEmbeddingsF32(
            full_equivalent,
            non_finite_embedding.data(),
            non_finite_embedding.size(),
            1,
            embedding_state.data(),
            embedding_logits.data(),
            embedding_logits.size(),
            workspace.data(),
            workspace.size()) ==
        vn97::LanguageStatus::kNonFinite);

    fixture.model.model_id[0] = 0;
    fixture.model.model_id[31] = 0;
    assert(
        vn97::ValidateLanguageModel(fixture.model) ==
        vn97::LanguageStatus::kInvalidModel);

    return 0;
}
