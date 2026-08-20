#include "engine/scheduler.hpp"
#include <algorithm>

namespace lyapunov {
namespace engine {

ContinuousScheduler::ContinuousScheduler(
    const SchedulerConfig& config,
    std::shared_ptr<BlockSpaceManager> block_manager
) : config_(config), block_manager_(block_manager) {}

void ContinuousScheduler::add_sequence(SequencePtr seq) {
    seq->set_status(SequenceStatus::WAITING);
    waiting_.push_back(seq);
}

void ContinuousScheduler::abort_sequence(int64_t seq_id) {
    for (auto it = running_.begin(); it != running_.end(); ++it) {
        if ((*it)->get_seq_id() == seq_id) {
            (*it)->set_status(SequenceStatus::FINISHED_ABORTED);
            block_manager_->free(*it);
            running_.erase(it);
            return;
        }
    }

    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
        if ((*it)->get_seq_id() == seq_id) {
            (*it)->set_status(SequenceStatus::FINISHED_ABORTED);
            waiting_.erase(it);
            return;
        }
    }
}

bool ContinuousScheduler::has_unfinished_sequences() const {
    return !waiting_.empty() || !running_.empty() || !swapped_.empty();
}

SchedulerOutputs ContinuousScheduler::schedule() {
    SchedulerOutputs outputs;

    // 1. If we have running sequences, perform a decode iteration step
    if (!running_.empty()) {
        outputs.is_prefill = false;
        std::vector<SequencePtr> scheduled;
        scheduled.reserve(running_.size());

        std::vector<SequencePtr> preempted;

        for (auto& seq : running_) {
            if (block_manager_->can_append_slot(seq)) {
                block_manager_->append_slot(seq);
                scheduled.push_back(seq);
            } else {
                seq->set_status(SequenceStatus::WAITING);
                block_manager_->free(seq);
                preempted.push_back(seq);
            }
        }

        for (auto it = preempted.rbegin(); it != preempted.rend(); ++it) {
            waiting_.push_front(*it);
        }

        running_ = scheduled;
        outputs.scheduled_seqs = scheduled;
        outputs.num_batched_tokens = static_cast<int>(scheduled.size());
        return outputs;
    }

    // 2. If no running sequences, schedule prefill requests from waiting queue
    outputs.is_prefill = true;
    int token_budget = config_.max_num_batched_tokens;
    int seq_budget = config_.max_num_seqs;

    while (!waiting_.empty() && seq_budget > 0) {
        SequencePtr seq = waiting_.front();
        int prompt_len = static_cast<int>(seq->get_prompt_len());

        if (prompt_len > token_budget) {
            break;
        }

        if (!block_manager_->can_allocate(seq)) {
            break;
        }

        waiting_.pop_front();
        block_manager_->allocate(seq);
        seq->set_status(SequenceStatus::RUNNING);

        running_.push_back(seq);
        outputs.scheduled_seqs.push_back(seq);

        token_budget -= prompt_len;
        seq_budget -= 1;
        outputs.num_batched_tokens += prompt_len;
    }

    return outputs;
}

void ContinuousScheduler::post_step(
    const std::vector<SequencePtr>& seqs,
    const std::vector<int32_t>& next_tokens
) {
    if (seqs.size() != next_tokens.size()) {
        throw std::runtime_error("Size mismatch between scheduled seqs and next tokens");
    }

    std::vector<SequencePtr> surviving_running;
    surviving_running.reserve(seqs.size());

    for (size_t i = 0; i < seqs.size(); ++i) {
        SequencePtr seq = seqs[i];
        int32_t token = next_tokens[i];

        seq->append_token_id(token);
        seq->set_num_computed_tokens(seq->get_total_len());

        const auto& sampling = seq->get_sampling_params();
        bool is_eos = (token == config_.eos_token_id) && !sampling.ignore_eos;
        bool is_length_capped = (static_cast<int>(seq->get_output_len()) >= sampling.max_tokens);

        if (is_eos) {
            seq->set_status(SequenceStatus::FINISHED_STOPPED);
            block_manager_->free(seq);
        } else if (is_length_capped) {
            seq->set_status(SequenceStatus::FINISHED_LENGTH_CAPPED);
            block_manager_->free(seq);
        } else {
            surviving_running.push_back(seq);
        }
    }

    running_ = surviving_running;
}

} // namespace engine
} // namespace lyapunov
