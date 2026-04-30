class VigorAuxiliary:
    def __init__(
        self,
        *,
        teacher_temperature: float,
        topk_prior: int,
        topk_final: int,
        random_k: int,
    ) -> None:
        ...

    def write_step(
        self,
        *,
        buffer: RolloutBuffer,
        t: int,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        state: State,
        step_out: PolicyStepOutput,
        active: torch.Tensor,
        remaining_budget: torch.Tensor,
        current_reward: TerminalRewardOutput,
    ) -> None:
        ...