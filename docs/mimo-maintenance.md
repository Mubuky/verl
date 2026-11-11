# MiMo maintenance branch

This branch starts at `2bc6ac46043e31855cea194236ff7548c91817c8`, the verl
revision pinned by Uni-Agent main `78348025ac6a5383ad62f17f78e67e943472caef`.
It adds the missing MiMo dataset and runner recipe while retaining native verl
training, sampling, masking, and optimization.

The reviewed MiMo source is `79090e58195e80a9cfe251c95ae7a78a2132b6e2`.
Its common ancestor with the selected base is
`2781c1e405fa80871a838bfdf9495b412df19c35`. The MiMo production/test delta
contains 163 files under `verl`, `recipes`, and `tests`; that delta was checked
against the selected base instead of replacing newer code with the older fork.

## Capability audit

| Current training requirement | Disposition |
| --- | --- |
| GRPO without group standard-deviation normalization | Native `core_algos` and configuration. |
| Multiple contexts per rollout | Native `compute_advantage_for_multi_trajectories` scores the final context once per session and broadcasts its advantage. |
| Token-mean loss and zero-mask padding | Native loss aggregation and `padding_utils`. |
| Asynchronous colocated training, stale-group drop/refill | Native V1 trainer and replay buffer. |
| Rollout/old-policy Token-TIS correction | Native `rollout_corr_helper`; no new estimator. |
| Canonical reward-based group filtering | Already implemented by the selected base; disabled in the integration recipe. |
| MiMo instance JSON rows | Port `MimoAgentSWEDataset`; preserve task payloads until rollout. |
| MiMo agent/environment lifecycle | Port the runner; retain the pinned mimoagent factory, setup, run, grading and cleanup. |
| Reward delivery | Return native Uni-Agent `TaskResult`; remove reward POST, retries, and completion pings. |
| Harness selection | Native runner kwargs select one `harness_config`; multiple runners use Uni-Agent's own dispatch. No duplicated labels or legacy mixed-harness environment aliases. |
| Model/transport failures | Raise before grading and keep cleanup; ordinary zero/partial rewards remain valid results. |

MiMo's tool/repetition advantage penalties, deep-failure masks, signed balancing,
harness-specific advantage subgroups, web-design rewards, ARVO and general-task
recipes, Megatron-only padding and SGLang metrics are outside this recipe's
active requirements. They are not added as unused training paths. No trainer,
optimizer, rollout engine or algorithm implementation has been replaced.

Platform scheduling, sandbox deployment and application scoring remain in the
consumer. The recipe accepts environment overrides and invokes mimoagent's
existing environment factory; it contains no platform resource identifiers.

## Runner contract

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        agent_runners:
          mimoagent:
            runner_fqn: recipes.code.mimoagent_runner.mimoagent_runner
            runner_kwargs:
              harness_config: /path/to/harness.yaml
            dispatch_mode: ray_task
            max_concurrent_sessions: 0
            trajectory_selection: all
```

The runner returns `TaskResult(reward=..., finished=True, extra_info=...)` after
successful grading. `agent_completed`, `agent_status`, tool errors and verifier
metadata remain in `extra_info`. A truncated but gradable episode can retain its
partial reward; `finished` records that the grading lifecycle completed, as in
the original MiMo contract. Infrastructure and model-query failures return no
training result.

The integration targets unchanged
`XiaomiMiMo/mimoagent@c3df8469f250b15e7906487a6d52be4c1827a509`.
The six runner-contract cases pass without a mimoagent fork. Joint tests use
Uni-Agent code `60f443808a6099fb91ee1ac88ae9f710e67f826f` and this recipe at
`d799866371163b0762302f2a7e5a1f09e15ee9d8`. Uni-Agent `710a0bf` pins this
revision and has the same Python source tree.

## Validation

The isolated Python 3.12.12 runtime uses Torch 2.11.0, vLLM 0.20.2,
Transformers 5.10.4, Ray 2.58.0 and TransferQueue 0.1.11. All 244 registry
dependencies and component import paths match the consumer's frozen lock.
Existing FSDP/offload and vLLM staged-wake fixes stay in the consumer. The newer
Torch 2.13.0/vLLM 0.29.0 runtime could not be installed from the remote wheel
sources and remains unverified.

- Native multi-trajectory GRPO, loss aggregation, Token-TIS, padding,
  ReplayBuffer and a real TransferQueue 0.1.11 service: 95 passed.
- Uni-Agent gateway/framework, including HTTP/SSE and real Ray actors:
  717 passed, 9 optional dependency skips, 17 external level1 probes deselected.
- MiMo runner and consumer contract/monitoring tests: 36 passed, 1 optional
  Docker-image fixture skipped.

An isolated four-H200 trial with Codex's default compaction threshold completed
one native update. Two groups of eight rollouts were consumed, with rewards 6/8
and 1/8 and no padding. Gradient norm was `0.28590438`; 202,871,959 elements
changed in the native BF16 HF export, with L2 delta `0.02075249`. The optimizer
state was 75,281,850,592 bytes. Rollout/training probability correlation was
`0.99843889`; Token-TIS mean weight was `0.98054516` and effective sample size
`0.99796887`. All 29 completed sessions passed generation-span, mask and logprob
checks; three unfinished prefetch sessions remained ungraded. The test job and
sandbox resources were released. This establishes one effective parameter
update, not learning benefit or production-scale performance.

A separate forced-compaction trial failed because Uni-Agent main rejects
`parallel_tool_calls=false` on Codex's empty-tool compaction request. Failed
sessions remained ungraded and produced no optimizer update. Live compaction
and optimizer consumption of compacted multi-chain trajectories remain switching
requirements. No verl algorithm or mimoagent compatibility patch was added.
