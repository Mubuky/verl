# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a configured MimoAgent harness through a Uni-Agent session.

Uni-Agent owns the model gateway and TransferQueue capture. MimoAgent owns the
selected agent loop, task environment, and grading. This bridge adapts those
contracts and returns the native Uni-Agent TaskResult.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import yaml
from uni_agent.tasks import TaskResult

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)

_MODEL_BACKED_AGENT_TYPES = {"default", "bashonly-agent", "mimocode-agent", "cc-agent", "codex-agent"}
_UNGRADABLE_AGENT_STATUSES = {"InfraError", "ModelQueryError", "CodexError"}


@dataclass
class _TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def _load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"harness config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError(f"harness config must be a mapping: {config_path}")
    return config


def _extract_task(raw_prompt: Any, instance: dict[str, Any]) -> str:
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt
    if isinstance(raw_prompt, list):
        for message in raw_prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return str(instance.get("problem_statement") or "")


def _build_model(config: dict[str, Any], gateway_url: str, *, agent_type: str):
    model_config = dict(config.get("model") or {})
    model_name = str(model_config.get("model_name") or "policy")
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    gateway_url = gateway_url.rstrip("/")
    if agent_type == "claude-code":
        gateway_url = gateway_url.removesuffix("/v1")
    model_kwargs["base_url"] = gateway_url
    model_kwargs["api_key"] = "not-needed"
    model_config.update(model_name=model_name, model_kwargs=model_kwargs)

    if agent_type in _MODEL_BACKED_AGENT_TYPES:
        from mimoagent.models import get_model

        return get_model(config=model_config)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name, model_kwargs=model_kwargs),
        token_stats=_TokenStats(),
        n_calls=0,
    )


def _apply_agent_model_override(
    agent_type: str,
    agent_config: dict[str, Any],
    model: SimpleNamespace,
) -> None:
    """Apply Codex's catalog model name without duplicating the policy config."""
    if agent_type != "codex":
        return
    model_name = agent_config.pop("model_name", None)
    if model_name is not None:
        model.config.model_name = str(model_name)


def _session_agent_msg_path(session: SessionHandle) -> Path | None:
    """Where this session's agent streams its messages, or None to not stream them.

    Handed to the agent as ``msg_path``, so a rollout that dies mid-way still leaves its
    partial messages on disk. The candidate list exists because Uni-Agent has written the
    session directory both at the log root and under a ``step_*`` level.
    """
    log_dir = os.getenv("UNI_AGENT_LOG_DIR")
    if not log_dir:
        return None
    log_root = Path(log_dir).expanduser()
    candidates = [log_root / session.session_id]
    candidates.extend(sorted(log_root.glob(f"step_*/{session.session_id}")))
    run_dir = next((path for path in candidates if path.is_dir()), None)
    if run_dir is None:
        run_dir = log_root / "agent_logs" / session.session_id
    return run_dir / "agent_msgs" / "main.log"


def _prepare_swebench_import_path() -> None:
    """Prefer the prebuilt-image-aware SWE-bench fork used by the old eval."""
    base = os.getenv("SWEBENCH_BASE")
    if not base:
        return
    source = str(Path(base).expanduser().resolve())
    if source not in sys.path:
        sys.path.insert(0, source)

    import swebench

    module_path = Path(swebench.__file__).resolve()
    if Path(source) not in module_path.parents:
        raise RuntimeError(f"SWE-bench source mismatch: expected under {source}, imported {module_path}")
    logger.info("SWE-bench source=%s", module_path)


def _run_sync(
    *,
    raw_prompt: Any,
    instance: dict[str, Any],
    session: SessionHandle,
    config: dict[str, Any],
    agent_overrides: dict[str, Any],
    environment_overrides: dict[str, Any],
) -> dict[str, Any]:
    _prepare_swebench_import_path()
    from mimoagent.agents.factory import get_agent_class
    from mimoagent.environments.utils import make_dataset_env

    environment_config = dict(config.get("environment") or {})
    environment_config.update(environment_overrides)
    environment = make_dataset_env(instance, **environment_config)
    try:
        environment.setup_environment()
        agent_config = dict(config.get("agent") or {})
        agent_type = str(agent_config.pop("type", "default"))
        agent_config.update(agent_overrides)
        model = _build_model(config, session.base_url or "", agent_type=agent_type)
        _apply_agent_model_override(agent_type, agent_config, model)
        agent_cls = get_agent_class(agent_type)
        msg_path = _session_agent_msg_path(session)
        if msg_path is not None:
            agent_config["msg_path"] = msg_path
        task = _extract_task(raw_prompt, instance)
        prompt_prefix = config.get("prompt_prefix")
        if prompt_prefix:
            if not isinstance(prompt_prefix, str):
                raise ValueError("prompt_prefix must be a string when configured")
            task = f"{prompt_prefix.rstrip()}\n\n--- Task ---\n{task}"
        agent = agent_cls(model, environment.env, **agent_config)
        status, result = agent.run(task)
        agent_completed = status == agent_cls.IDLE_STATUS
        if status in _UNGRADABLE_AGENT_STATUSES:
            raise RuntimeError(f"{agent_type} rollout failed with status={status}: {str(result)[-500:]}")
        reward, test_output, reward_extra_info = environment.calculate_reward()
        reward_info = {
            **(reward_extra_info or {}),
            "reward": float(reward),
            "finished": True,
            "agent_type": agent_type,
            "agent_status": status,
            "agent_completed": agent_completed,
            "termination_kind": "completed" if agent_completed else "truncated",
            "result": result[-5000:] if isinstance(result, str) else str(result),
            "test_output": test_output[-5000:] if isinstance(test_output, str) else str(test_output),
        }
        agent_error_flags = getattr(agent, "tool_call_errors", None)
        if isinstance(agent_error_flags, list | tuple) and agent_error_flags:
            normalized_error_flags = [bool(value) for value in agent_error_flags]
            reward_info["tool_call_error_flags"] = normalized_error_flags
            reward_info["tool_call_error_flag_source"] = "agent_step"
        else:
            normalized_error_flags = []
            reward_info["tool_call_error_flag_source"] = "unavailable"
        tool_error_counts = getattr(agent, "tool_call_error_counts", None)
        tool_error_names = getattr(agent, "tool_call_error_names", None)
        reward_info["tool_call_error_types"] = {
            str(key): int(value) for key, value in (tool_error_counts or {}).items()
        }
        aggregate_tool_error_count = int(sum((tool_error_counts or {}).values()))
        if not aggregate_tool_error_count and normalized_error_flags:
            aggregate_tool_error_count = int(sum(normalized_error_flags))
        reward_info["tool_call_error_count"] = aggregate_tool_error_count
        reward_info["codex_transport_error_count"] = int(getattr(agent, "codex_transport_error_count", 0))
        for error_type in (
            "unknown_tool",
            "invalid_arguments",
            "incompatible_payload",
            "other_tool_error",
        ):
            reward_info[f"tool_call_error_{error_type}_count"] = int((tool_error_counts or {}).get(error_type, 0))
        reward_info["tool_call_error_names"] = {str(key): int(value) for key, value in (tool_error_names or {}).items()}
        if agent_type == "claude-code":
            reward_info["claude_code_status"] = status
        return reward_info
    finally:
        environment.cleanup()


async def mimoagent_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    harness_config: str,
    tools_kwargs: dict | None = None,
    **runner_kwargs,
) -> TaskResult:
    """Run the selected MimoAgent harness and return its graded outcome."""
    tools_kwargs = dict(tools_kwargs or {})
    instance = tools_kwargs.get("instance")
    if not isinstance(instance, dict):
        raise ValueError(f"sample {sample_index} is missing tools_kwargs.instance")
    if not session.base_url:
        raise ValueError(f"sample {sample_index} has no Uni-Agent gateway base_url")

    config_path = Path(harness_config).expanduser().resolve()
    config = _load_config(config_path)
    agent_overrides = dict(runner_kwargs.pop("agent_overrides", {}) or {})
    environment_overrides = dict(runner_kwargs.pop("environment_overrides", {}) or {})
    reward_info = await asyncio.to_thread(
        _run_sync,
        raw_prompt=raw_prompt,
        instance=instance,
        session=session,
        config=config,
        agent_overrides=agent_overrides,
        environment_overrides=environment_overrides,
    )
    reward_info["dataset_index"] = int(tools_kwargs.get("dataset_index", sample_index))
    reward_info["harness_config_path"] = str(config_path)
    logger.info(
        "MimoAgent harness=%s sample=%s reward=%s status=%s tool_call_errors=%s",
        reward_info.get("agent_type"),
        sample_index,
        reward_info["reward"],
        reward_info.get("agent_status"),
        reward_info.get("tool_call_error_count", 0),
    )

    return TaskResult(
        reward=reward_info.pop("reward"),
        finished=reward_info.pop("finished"),
        extra_info=reward_info,
    )
