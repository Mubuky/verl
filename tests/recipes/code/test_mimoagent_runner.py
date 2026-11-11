"""MiMo environment lifecycle through the native Uni-Agent runner contract."""

from types import SimpleNamespace

import pytest
import yaml
from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks import TaskResult

from recipes.code import mimoagent_runner as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("status,reward", [("Idle", 1.0), ("Idle", 0.0), ("LimitsExceeded", 0.25)])
async def test_native_result_preserves_partial_rewards_and_cleanup(monkeypatch, tmp_path, status, reward):
    events = []

    class Environment:
        env = object()

        def setup_environment(self):
            events.append("setup")

        def calculate_reward(self):
            events.append("grade")
            return reward, "verifier output", {"verifier": "ok"}

        def cleanup(self):
            events.append("cleanup")

    class Agent:
        IDLE_STATUS = "Idle"

        def __init__(self, model, env, **kwargs):
            assert model.config.model_kwargs["base_url"] == "http://gateway/sessions/test/v1"
            self.tool_call_errors = [False, True]

        def run(self, task):
            events.append("agent")
            assert task == "task text"
            return status, "done"

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: Agent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: Environment())
    monkeypatch.delenv("UNI_AGENT_LOG_DIR", raising=False)
    path = tmp_path / "harness.yaml"
    path.write_text(yaml.safe_dump({"agent": {"type": "codex"}, "model": {"model_name": "policy"}}))
    result = await runner.mimoagent_runner(
        raw_prompt="task text",
        session=SessionHandle("test", base_url="http://gateway/sessions/test/v1"),
        sample_index=0,
        harness_config=str(path),
        tools_kwargs={"instance": {"docker_image": "fixture"}},
    )
    assert isinstance(result, TaskResult)
    assert result.reward == reward and result.finished is True
    assert result.extra_info["agent_completed"] == (status == "Idle")
    assert result.extra_info["tool_call_error_flags"] == [False, True]
    assert events == ["setup", "agent", "grade", "cleanup"]


@pytest.mark.parametrize("status", ["InfraError", "ModelQueryError", "CodexError"])
def test_model_and_transport_failures_remain_unscored(monkeypatch, status):
    events = []

    class Environment:
        env = object()

        def setup_environment(self):
            events.append("setup")

        def calculate_reward(self):
            pytest.fail("failed rollout must not be graded")

        def cleanup(self):
            events.append("cleanup")

    class Agent:
        IDLE_STATUS = "Idle"

        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            return status, "transport failed"

    monkeypatch.setattr("mimoagent.agents.factory.get_agent_class", lambda _: Agent)
    monkeypatch.setattr("mimoagent.environments.utils.make_dataset_env", lambda *args, **kwargs: Environment())
    monkeypatch.delenv("UNI_AGENT_LOG_DIR", raising=False)
    with pytest.raises(RuntimeError, match=status):
        runner._run_sync(
            raw_prompt="task text",
            instance={},
            session=SimpleNamespace(base_url="http://gateway/v1"),
            config={"agent": {"type": "codex"}},
            agent_overrides={},
            environment_overrides={},
        )
    assert events == ["setup", "cleanup"]
