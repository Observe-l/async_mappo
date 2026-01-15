from __future__ import annotations

import argparse
from collections import defaultdict

from onpolicy.envs.rul_schedule.schedule import async_scheduling


def _build_fixed_policy_maps(num_agents: int):
    """Return (empty_action_options_by_agent, loaded_action_by_agent).

    Mapping is defined for agents 0-11. For other agent counts, we either
    use a subset (num_agents < 12) or raise (num_agents > 12).
    """

    if num_agents > 12:
        raise ValueError(
            f"Fixed policy mapping is only defined for 12 agents (0-11); got num_agents={num_agents}."
        )

    empty_action_groups = {
        (0, 1, 2): list(range(0, 9)),
        (3, 4, 5): list(range(9, 18)),
        (6, 7): list(range(18, 27)),
        (8, 9): list(range(27, 36)),
        (10, 11): list(range(36, 45)),
    }
    loaded_action_groups = {
        (0, 1, 2): 49,
        (3, 4, 5): 48,
        (6, 7): 47,
        (8, 9): 46,
        (10, 11): 45,
    }

    empty_by_agent: dict[int, list[int]] = {}
    loaded_by_agent: dict[int, int] = {}

    for agent_ids, actions in empty_action_groups.items():
        for agent_id in agent_ids:
            empty_by_agent[agent_id] = actions
    for agent_ids, action in loaded_action_groups.items():
        for agent_id in agent_ids:
            loaded_by_agent[agent_id] = action

    # Trim for smaller agent counts
    empty_by_agent = {i: empty_by_agent[i] for i in range(num_agents) if i in empty_by_agent}
    loaded_by_agent = {i: loaded_by_agent[i] for i in range(num_agents) if i in loaded_by_agent}

    return empty_by_agent, loaded_by_agent


class FixedPolicy:
    """Deterministic fixed policy.

    If weight == 0: pick from the agent's action set using round-robin.
    Else: pick the agent's fixed (loaded) action.
    """

    def __init__(self, num_agents: int):
        self.empty_by_agent, self.loaded_by_agent = _build_fixed_policy_maps(num_agents)
        self._rr_counters: dict[int, int] = defaultdict(int)

    def action(self, agent_id: int, weight) -> int:
        if agent_id not in self.empty_by_agent or agent_id not in self.loaded_by_agent:
            raise ValueError(
                f"No fixed-policy mapping for agent_id={agent_id}. "
                "This script expects agent ids 0-11."
            )

        w = float(weight)
        if w <= 0.0:
            options = self.empty_by_agent[agent_id]
            idx = self._rr_counters[agent_id] % len(options)
            self._rr_counters[agent_id] += 1
            return int(options[idx])

        return int(self.loaded_by_agent[agent_id])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='fixed_schedule', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=12, help="number of trucks")
    parser.add_argument('--use_rul_agent', default=True, action='store_true', help="Use agent to predict RUL")
    parser.add_argument('--rul_threshold', default=7, type=int, help="RUL threshold")
    parser.add_argument('--rul_state', default=True, action='store_true', help="Include RUL in observations")
    all_args = parser.parse_args()

    env = async_scheduling(all_args)
    policy = FixedPolicy(num_agents=all_args.num_agents)
    obs = env.reset()
    done = False
    while not done:
        action_dict = {}
        for agent_id in obs.keys():
            agent_id_int = int(agent_id)
            truck = env.truck_agents[agent_id_int]
            action_dict[agent_id_int] = policy.action(agent_id_int, truck.weight)

        obs, rewards, dones, infos = env.step(action_dict)
        done = all(dones)