#include "language_fixture.h"

#include "vn97/runtime.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    vn97_test::TinyLanguageFixture fixture;

    vn97::RuntimeConfig config;
    config.layers =
        vn97_test::TinyLanguageFixture::kLayers;
    config.batch = 1;
    config.d_model =
        vn97_test::TinyLanguageFixture::kModel;
    config.d_state =
        vn97_test::TinyLanguageFixture::kState;
    config.recurrent_backend =
        vn97::RecurrentBackend::kScalar;
    config.packed_backend =
        vn97::PackedTernaryBackend::kScalar;

    vn97::RuntimeSession* session = nullptr;
    assert(
        vn97::RuntimeSession::Create(
            config,
            &session) ==
        vn97::RuntimeStatus::kOk);
    assert(session != nullptr);
    assert(
        session->Activate() ==
        vn97::RuntimeStatus::kOk);

    std::vector<float> logits(
        vn97_test::TinyLanguageFixture::kVocab,
        0.0f);
    const std::uint32_t token = 1;
    assert(
        session->InferStep(
            fixture.model,
            &token,
            logits.data(),
            logits.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(
        session->Info().sequence_position ==
        1);

    bool bound = false;
    std::array<std::uint8_t, 32> model_id = {};
    assert(
        session->ModelBinding(
            &bound,
            model_id.data(),
            model_id.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(bound);
    assert(
        std::equal(
            model_id.begin(),
            model_id.end(),
            fixture.model.model_id));

    assert(
        session->Advance(1) ==
        vn97::RuntimeStatus::kInferenceError);

    assert(
        session->Suspend() ==
        vn97::RuntimeStatus::kOk);

    std::size_t checkpoint_size = 0;
    assert(
        session->CheckpointSize(
            &checkpoint_size) ==
        vn97::RuntimeStatus::kOk);
    assert(
        checkpoint_size ==
        100 +
            vn97_test::TinyLanguageFixture::kModel *
                vn97_test::TinyLanguageFixture::kState *
                sizeof(float));

    std::vector<std::uint8_t> checkpoint(
        checkpoint_size);
    std::size_t written = 0;
    assert(
        session->WriteCheckpoint(
            checkpoint.data(),
            checkpoint.size(),
            &written) ==
        vn97::RuntimeStatus::kOk);
    assert(written == checkpoint.size());

    const std::array<std::uint8_t, 8> run2 =
        {'V','N','9','7','R','U','N','2'};
    assert(
        std::equal(
            run2.begin(),
            run2.end(),
            checkpoint.begin()));

    delete session;
    session = nullptr;

    assert(
        vn97::RuntimeSession::Restore(
            checkpoint.data(),
            checkpoint.size(),
            &session) ==
        vn97::RuntimeStatus::kOk);
    assert(session != nullptr);
    assert(
        session->Info().lifecycle ==
        vn97::RuntimeLifecycle::kSuspended);

    model_id.fill(0);
    bound = false;
    assert(
        session->ModelBinding(
            &bound,
            model_id.data(),
            model_id.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(bound);
    assert(
        std::equal(
            model_id.begin(),
            model_id.end(),
            fixture.model.model_id));

    std::vector<float> before(
        vn97_test::TinyLanguageFixture::kModel *
            vn97_test::TinyLanguageFixture::kState);
    assert(
        session->ReadState(
            before.data(),
            before.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(
        session->Resume() ==
        vn97::RuntimeStatus::kOk);

    auto wrong_model = fixture.model;
    wrong_model.model_id[31] ^= 0x7f;
    assert(
        session->InferStep(
            wrong_model,
            &token,
            logits.data(),
            logits.size()) ==
        vn97::RuntimeStatus::kModelMismatch);
    assert(
        session->Info().sequence_position ==
        1);

    assert(
        session->Suspend() ==
        vn97::RuntimeStatus::kOk);
    std::vector<float> after(before.size());
    assert(
        session->ReadState(
            after.data(),
            after.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(before == after);

    assert(
        session->Resume() ==
        vn97::RuntimeStatus::kOk);
    assert(
        session->InferStep(
            fixture.model,
            &token,
            logits.data(),
            logits.size()) ==
        vn97::RuntimeStatus::kOk);
    assert(
        session->Info().sequence_position ==
        2);

    assert(
        session->Suspend() ==
        vn97::RuntimeStatus::kOk);
    delete session;
    session = nullptr;

    auto tampered = checkpoint;
    tampered[64] ^= 1;
    vn97::RuntimeSession* rejected = nullptr;
    assert(
        vn97::RuntimeSession::Restore(
            tampered.data(),
            tampered.size(),
            &rejected) ==
        vn97::RuntimeStatus::kCheckpointCorrupt);
    assert(rejected == nullptr);

    vn97::RuntimeSession* legacy = nullptr;
    assert(
        vn97::RuntimeSession::Create(
            config,
            &legacy) ==
        vn97::RuntimeStatus::kOk);
    assert(
        legacy->Activate() ==
        vn97::RuntimeStatus::kOk);
    assert(
        legacy->Advance(1) ==
        vn97::RuntimeStatus::kOk);
    assert(
        legacy->Suspend() ==
        vn97::RuntimeStatus::kOk);

    std::size_t legacy_size = 0;
    assert(
        legacy->CheckpointSize(
            &legacy_size) ==
        vn97::RuntimeStatus::kOk);
    std::vector<std::uint8_t> legacy_checkpoint(
        legacy_size);
    assert(
        legacy->WriteCheckpoint(
            legacy_checkpoint.data(),
            legacy_checkpoint.size(),
            &written) ==
        vn97::RuntimeStatus::kOk);
    const std::array<std::uint8_t, 8> run1 =
        {'V','N','9','7','R','U','N','1'};
    assert(
        std::equal(
            run1.begin(),
            run1.end(),
            legacy_checkpoint.begin()));
    delete legacy;
    legacy = nullptr;

    assert(
        vn97::RuntimeSession::Restore(
            legacy_checkpoint.data(),
            legacy_checkpoint.size(),
            &legacy) ==
        vn97::RuntimeStatus::kOk);
    assert(
        legacy->Resume() ==
        vn97::RuntimeStatus::kOk);
    assert(
        legacy->InferStep(
            fixture.model,
            &token,
            logits.data(),
            logits.size()) ==
        vn97::RuntimeStatus::kModelMismatch);

    delete legacy;
    return 0;
}
