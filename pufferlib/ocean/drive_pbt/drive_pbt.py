import os
import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.scenario_log import (
    append_scenario_logs,
    resolve_scenario_log_path,
    split_aggregate_and_scenario,
)

_DRIVE_INI = "pufferlib/config/ocean/drive.ini"
_PARTNER_REL_SCALE = 0.02  # drive.h: rel_xy stored as meters * 0.02
from pufferlib.ocean.drive_pbt.agent_sampler import AgentSampler
from pufferlib.ocean.drive_pbt.curriculum_sampler import (
    CurriculumSampler,
    load_difficulty_types,
)


def generate_map_policy_assignments(global_ids, num_policies, num_assignments, seed=1):
    """Generate entity-to-policy assignment candidates for each map.
    """
    global_ids = np.asarray(global_ids)
    num_policies = int(num_policies)
    num_assignments = int(num_assignments)

    rng = np.random.default_rng(seed)
    result = []
    for map_id in range(global_ids.shape[0]):
        entity_ids = np.flatnonzero(global_ids[map_id] >= 0).astype(np.int64)
        num_entities = int(entity_ids.size)
        assignments = np.empty((num_assignments, num_entities), dtype=np.int32)
        for assignment_id in range(num_assignments):
            assignments[assignment_id] = rng.choice(
                num_policies, size=num_entities, replace=True
            )
        result.append((entity_ids, assignments))
    return result


class Drive_PBT(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        reward_head_diff=0.0,
        reward_lane_dist=0.0,
        aggressive_speed=0.0,
        reward_speed=0.0,
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        map_dir="resources/drive/binaries/training",
        sequential_map_sampling=False,
        pbt_mode="reactive", # reactive or replay
        population_path=None, # for replay
        ego_ratio=0.0, # for replay
        agent_sampling=False,
        strategy="prioritized",
        curriculum_types=None,
        curriculum_types_path=None,
        curriculum_steps=10000,
        score_transform="power",
        num_policy_assignments=50,
        policy_assignment_seed=1,

        scenario_log_path=None,
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.scenario_log_path = resolve_scenario_log_path(scenario_log_path, _DRIVE_INI)
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.aggressive_speed = aggressive_speed
        self.reward_head_diff = reward_head_diff
        self.reward_lane_dist = reward_lane_dist
        self.reward_speed = reward_speed
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        # Observation space calculation
        self.ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            dynamics_model
        )

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.map_dir = map_dir

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. Got: {self.control_mode_str}"
            )
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists
        binary_path = f"{map_dir}/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        self.max_controlled_agents = int(max_controlled_agents)

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs, ego_indices = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            ego_ratio=ego_ratio,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            sequential_map_sampling=sequential_map_sampling,
        )
        # agent_offsets[-1] works in both cases, just making it explicit that num_agents is ignored if sequential_map_sampling is True
        self.num_agents = num_agents if not sequential_map_sampling else agent_offsets[-1]
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        self.ego_indices = np.asarray(ego_indices, dtype=np.int64)
        ao = np.asarray(agent_offsets, dtype=np.int64)
        self.num_ego_per_env = [int(np.sum((self.ego_indices >= ao[i]) & (self.ego_indices < ao[i + 1]))) for i in range(num_envs)]
        self.other_mask = np.ones(self.num_agents, dtype=bool)
        self.other_mask[self.ego_indices] = False
        # Canonical non-ego agents (ascending global index). Slot i <-> other_indices_arr[i].
        self.other_indices_arr = np.flatnonzero(self.other_mask).astype(np.int64)
        self._ego_index_set = set(self.ego_indices.tolist())
        super().__init__(buf=buf)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            ego_local = (self.ego_indices - cur)[(self.ego_indices >= cur) & (self.ego_indices < nxt)].astype(np.int32).tolist()
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                aggressive_speed=aggressive_speed,
                reward_lane_dist=reward_lane_dist,
                reward_head_diff=reward_head_diff,
                reward_speed=reward_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                num_ego=self.num_ego_per_env[i],
                ego_local_indices=ego_local,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
                scenario_log_path=self.scenario_log_path or "",
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        self.ego_ratio = ego_ratio
        self.population_path = population_path
        self.pbt_mode = pbt_mode
        self.agent_sampling = agent_sampling
        if self.pbt_mode == "reactive" and not self.agent_sampling:
            raise ValueError("reactive map-policy-assignment PLR requires agent_sampling=True")
        saved_dir = os.path.join(self.population_path, "saved")
        fp_gid = os.path.join(saved_dir, "global_ids.npy")
        if self.agent_sampling:
            self.global_ids = np.load(fp_gid, mmap_mode="r")
            valid_global_ids = np.asarray(self.global_ids)[np.asarray(self.global_ids) >= 0]
            self.total_agents = int(valid_global_ids.max()) + 1 if valid_global_ids.size else 0
        if pbt_mode == "replay":
            fp_ao = os.path.join(saved_dir, "other_actions_agent_offsets.npy")
            fp_m = os.path.join(saved_dir, "other_actions_map_ids.npy")
            self.actions_agent_offsets = np.load(fp_ao, mmap_mode="r")[0] # TOOD: (10, 10001)으로 하는데, 그럴 필요 없음. 데이터 (10001,)으로 줄이기
            self.actions_map_id = np.load(fp_m, mmap_mode="r")[0] # TOOD: (10, 10000)으로 하는데, 그럴 필요 없음. 데이터 (10000,)으로 줄이기
            self.total_agents = int(self.actions_agent_offsets[-1])
            fp_actions = os.path.join(saved_dir, "other_actions_actions.npy")
            self.other_actions = np.load(fp_actions, mmap_mode="r")
            if not agent_sampling:
                self._allocate_replay(self.num_agents, self.map_ids)
            else:
                self.replay_actions = np.zeros(
                    (self.num_agents, self.resample_frequency, 1), dtype=np.int32
                )
            self._episode_return = np.zeros(self.num_agents, dtype=np.float32)
            n_other = int(self.other_indices_arr.size)
            self.minimum_distance = np.full(n_other, np.inf, dtype=np.float32)
            self.minimum_ego_idx = np.full(n_other, -1, dtype=np.int64)
            self.minimum_other_local_idx = np.full(n_other, -1, dtype=np.int64)
            self.minimum_other_global_idx = np.full(n_other, -1, dtype=np.int64)
            self.score_metric = np.zeros(n_other, dtype=np.float32)
            self.rollout_flatten = np.full(n_other, -1, dtype=np.int64)
            self._init_minimum_map_idx(self.map_ids)

            self.agent_sampler = self._make_agent_sampler(
                num_population=int(self.other_actions.shape[0]),
                strategy=strategy,
                score_transform=score_transform,
                num_assignments=self.num_maps,
                curriculum_types=curriculum_types,
                curriculum_types_path=curriculum_types_path,
                curriculum_steps=curriculum_steps,
            )
            self._last_sampling_metrics = {}
            self._last_raw_return_metrics = {}
        elif pbt_mode == "reactive":
            populations = sorted(
                f for f in os.listdir(population_path)
                if f.endswith(".pt")
            )
            self.num_other_policies = len(populations)
            n_other = int(self.other_indices_arr.size)
            self.policy_per_slot_flatten = np.full(n_other, -1, dtype=np.int64)
            self.policy_per_slot = [np.array([], dtype=np.int64) for _ in range(self.num_other_policies)]
            self.num_policy_assignments = int(num_policy_assignments)
            self.map_policy_assignments = generate_map_policy_assignments(
                self.global_ids,
                self.num_other_policies,
                self.num_policy_assignments,
                seed=policy_assignment_seed,
            )
            self.rollout_flatten = np.full(n_other, -1, dtype=np.int64)
            if self.agent_sampling:
                if self.total_agents < 1:
                    raise ValueError(f"total_agents must be >= 1, got {self.total_agents}")
                self._episode_return = np.zeros(self.num_agents, dtype=np.float32)
                n_other = int(self.other_indices_arr.size)
                self.minimum_distance = np.full(n_other, np.inf, dtype=np.float32)
                self.minimum_ego_idx = np.full(n_other, -1, dtype=np.int64)
                self.minimum_other_local_idx = np.full(n_other, -1, dtype=np.int64)
                self.minimum_other_global_idx = np.full(n_other, -1, dtype=np.int64)
                self.score_metric = np.zeros(n_other, dtype=np.float32)
                self.agent_sampler = self._make_agent_sampler(
                    num_population=self.num_policy_assignments,
                    strategy=strategy,
                    score_transform=score_transform,
                    num_assignments=self.num_maps,
                    curriculum_types=curriculum_types,
                    curriculum_types_path=curriculum_types_path,
                    curriculum_steps=curriculum_steps,
                )
                self._last_sampling_metrics = {}
                self._last_raw_return_metrics = {}

    def _make_agent_sampler(
        self,
        num_population,
        strategy,
        num_assignments,
        score_transform,
        curriculum_types=None,
        curriculum_types_path=None,
        curriculum_steps=10000,
    ):
        if strategy == "curriculum":
            types = load_difficulty_types(
                curriculum_types, curriculum_types_path, num_population
            )
            return CurriculumSampler(
                num_population=num_population,
                difficulty_types=types,
                curriculum_steps=curriculum_steps,
                pbt_mode=self.pbt_mode,
                num_assignments=num_assignments,
            )
        return AgentSampler(
            num_population=num_population,
            strategy=strategy,
            score_transform=score_transform,
            num_assignments=num_assignments,
        )

    def _init_minimum_map_idx(self, map_ids):
        map_ids = np.asarray(map_ids, dtype=np.int64).reshape(-1)
        n = int(map_ids.size)
        if not hasattr(self, "minimum_map_idx") or self.minimum_map_idx.shape[0] != n:
            self.minimum_map_idx = np.full(n, -1, dtype=np.int64)
        else:
            self.minimum_map_idx.fill(-1)
        np.copyto(self.minimum_map_idx, map_ids)

    def _reset_other_indices(self):
        """Episode/rollout start (after vec_reset): metrics, slot identity, LUT, policy assignment."""
        # init metrics
        self.minimum_distance.fill(np.inf)
        self.minimum_ego_idx.fill(-1)
        self.minimum_other_local_idx.fill(-1)
        self.minimum_other_global_idx.fill(-1)

        self._init_minimum_map_idx(self.map_ids)
        self.rollout_flatten.fill(-1)
        if self.pbt_mode == "replay":
            self.replay_actions.fill(-1)
        else:
            self.policy_per_slot_flatten.fill(-1)

        # assign other indices
        live_entity_ids = self.get_global_partner_state()["ego_id"].astype(np.int64)
        map_per_agent = self._map_per_agent()
        entity_ids = live_entity_ids[self.other_mask]
        map_ids = map_per_agent[self.other_mask]
        gid = self.global_ids
        valid = (
            (entity_ids >= 0)
            & (map_ids >= 0)
            & (map_ids < gid.shape[0])
            & (entity_ids < gid.shape[1])
        )
        self.minimum_other_local_idx[valid] = entity_ids[valid]
        self.minimum_other_global_idx[valid] = gid[map_ids[valid], entity_ids[valid]]

        # (env_i, local entity id) -> other slot; map_id alone collides across envs on same map
        max_entity = gid.shape[1]
        self._env_entity_to_other_slot = np.full((self.num_envs, max_entity), -1, dtype=np.int64)
        env_per_other = self._env_per_agent()[self.other_indices_arr]
        tracked = self.minimum_other_local_idx >= 0
        slots = np.flatnonzero(tracked)
        envs = env_per_other[tracked]
        entities = self.minimum_other_local_idx[tracked]
        in_bounds = (
            (envs >= 0)
            & (envs < self.num_envs)
            & (entities >= 0)
            & (entities < max_entity)
        )
        self._env_entity_to_other_slot[envs[in_bounds], entities[in_bounds]] = slots[in_bounds]
        flat, wandb_metrics = self.agent_sampler.sample(self.minimum_map_idx)
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        self._last_sampling_metrics = {k: float(v) for k, v in wandb_metrics.items()}
        self._last_sampling_metrics.update(self._last_raw_return_metrics)
        self._set_per_slot(flat)

    def _set_policy_per_slot(self, flat):
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        np.copyto(self.policy_per_slot_flatten, flat)
        self.policy_per_slot = [
            self.other_indices_arr[self.policy_per_slot_flatten == policy_idx].astype(np.int64, copy=False)
            for policy_idx in range(self.num_other_policies)
        ]

    def _set_per_slot(self, flat):
        """Expand one sampled record id per map environment to its agent slots."""
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        env_per_other = self._env_per_agent()[self.other_indices_arr]
        self.rollout_flatten[:] = flat[env_per_other]

        if self.pbt_mode == "reactive":
            self.map_policy_assignment_ids = flat.copy()
            self._set_reactive_per_slot(env_per_other)
        elif self.pbt_mode == "replay":
            self._set_replay_per_slot(flat)

    def _set_reactive_per_slot(self, env_per_other):
        """Resolve the selected map policy assignment for each live non-ego agent."""
        map_per_other = np.asarray(self.map_ids, dtype=np.int64)[env_per_other]
        entity_per_other = self.minimum_other_local_idx
        self.policy_per_slot_flatten.fill(-1)

        for slot in range(self.other_indices_arr.size):
            map_id = int(map_per_other[slot])
            entity_id = int(entity_per_other[slot])
            assignment_id = int(self.rollout_flatten[slot])
            entities, assignments = self.map_policy_assignments[map_id]
            pos = int(np.searchsorted(entities, entity_id))
            if pos >= entities.size or int(entities[pos]) != entity_id:
                continue
            self.policy_per_slot_flatten[slot] = int(assignments[assignment_id, pos])
        self._set_policy_per_slot(self.policy_per_slot_flatten)

    def _set_replay_per_slot(self, flat):
        """Copy the selected replay record trajectories into the action buffer."""
        agent_ind = 0
        for map_id, rollout_idx in zip(self.minimum_map_idx, flat):
            map_id = int(map_id)
            rollout_idx = int(rollout_idx)
            map_indices = np.where(self.actions_map_id == map_id)[0][0]
            agent_offsets = self.actions_agent_offsets[map_indices:map_indices + 2]
            num_agents_for_map = agent_offsets[1] - agent_offsets[0]
            if agent_ind + num_agents_for_map > self.num_agents:
                num_agents_for_map = self.num_agents - agent_ind
            self.replay_actions[agent_ind:agent_ind + num_agents_for_map] = self.other_actions[
                rollout_idx, agent_offsets[0]:agent_offsets[0] + num_agents_for_map
            ].copy()
            agent_ind += num_agents_for_map

    def _env_per_agent(self):
        ao = np.asarray(self.agent_offsets, dtype=np.int64)
        return np.searchsorted(ao[1:], np.arange(self.num_agents, dtype=np.int64), side="right")

    def _map_per_agent(self):
        return np.asarray(self.map_ids, dtype=np.int64)[self._env_per_agent()]

    def _metric_ego_global_indices(self):
        """Global flatten indices that C add_log counts in the ego_* bucket (matches drive.h)."""
        ao = np.asarray(self.agent_offsets, dtype=np.int64)
        ego = np.asarray(self.ego_indices, dtype=np.int64)
        c_ego = []
        for i in range(self.num_envs):
            cur, nxt = int(ao[i]), int(ao[i + 1])
            ego_local = (ego - cur)[(ego >= cur) & (ego < nxt)]
            if ego_local.size > 0:
                c_ego.extend((cur + ego_local).tolist())
            elif self.num_ego_per_env[i] > 0:
                c_ego.append(cur)
        return np.asarray(c_ego, dtype=np.int64)

    def _enrich_aggregate_metrics(self, aggregate):
        """Attach policy vs C-metric ego counts for dashboard / audit."""
        policy_ego_n = int(self.ego_indices.size)
        metric_ego = self._metric_ego_global_indices()
        metric_ego_n = int(metric_ego.size)
        aggregate["policy_ego_n"] = float(policy_ego_n)
        aggregate["metric_ego_n"] = float(metric_ego_n)
        legacy_n = int(np.setdiff1d(metric_ego, self.ego_indices, assume_unique=False).size)
        aggregate["legacy_ego_n"] = float(legacy_n)
        aggregate["other_policy_n"] = float(int(self.other_indices_arr.size))
        reported_ego_n = float(aggregate.get("ego_n", metric_ego_n))
        if reported_ego_n > 0 and legacy_n > 0:
            aggregate["ego_n_mismatch"] = float(reported_ego_n - policy_ego_n)

    def _partner_obs(self):
        start = self.ego_features
        stop = start + self.max_partner_objects * self.partner_features
        return self.observations[:, start:stop].reshape(
            self.num_agents, self.max_partner_objects, self.partner_features
        )

    def _update_minimum_distance(self):
        """
        input:
        dist: (num_agents, num_partners)
        other_ids: (num_agents, num_partners)
        map_ids: (num_agents, num_partners)
        output:
        minimum_distance: (num_others, )
        minimum_ego_idx: (num_other, )
        """
        if not self.agent_sampling:
            return
        partner_states = self.get_global_partner_state()
        other_ids = partner_states["other_id"].astype(np.int64)
        rel_xy = self._partner_obs()[:, :, :2]
        dist = np.linalg.norm(rel_xy, axis=-1) / _PARTNER_REL_SCALE
        env_per_agent = self._env_per_agent()
        env_ids = np.broadcast_to(env_per_agent[:, np.newaxis], other_ids.shape)
        lut = self._env_entity_to_other_slot

        ego_mask = np.zeros(self.num_agents, dtype=bool)
        ego_mask[self.ego_indices] = True
        valid = (
            ego_mask[:, np.newaxis]
            & (other_ids >= 0)
            & (env_ids >= 0)
            & (env_ids < lut.shape[0])
            & (other_ids < lut.shape[1])
        )
        if not np.any(valid):
            return

        other_slot = np.full(other_ids.shape, -1, dtype=np.int64)
        other_slot[valid] = lut[env_ids[valid], other_ids[valid]]
        valid &= other_slot >= 0
        if not np.any(valid):
            return

        ego_idx, slot = np.where(valid)
        flat_other = other_slot[ego_idx, slot]
        flat_dist = dist[ego_idx, slot]
        order = np.lexsort((flat_dist, flat_other)) # sort by distance and then by other id
        flat_other = flat_other[order]
        flat_dist = flat_dist[order]
        flat_ego = ego_idx[order]

        change = np.concatenate([[True], flat_other[1:] != flat_other[:-1]]) # detect change in other id
        best_other = flat_other[change]
        best_dist = flat_dist[change]
        best_ego = flat_ego[change]

        best = best_dist < self.minimum_distance[best_other]
        if not np.any(best):
            return
        self.minimum_distance[best_other[best]] = best_dist[best]
        self.minimum_ego_idx[best_other[best]] = best_ego[best]

    def _on_episode_end(self):
        tracked = (
            np.isfinite(self.minimum_distance)
            & (self.minimum_ego_idx >= 0)
            & (self.minimum_other_global_idx >= 0)
        )
        for other_slot in np.flatnonzero(tracked):
            ego_idx = int(self.minimum_ego_idx[other_slot])
            self.score_metric[other_slot] = self._episode_return[ego_idx]
        self._episode_return.fill(0.0)
        # Update Score
        map_per_other = self._map_per_agent()[self.other_indices_arr]
        raw_return_metrics = self.agent_sampler.update_policy_score(
            self.score_metric,
            self.minimum_other_global_idx,
            self.rollout_flatten,
            self.minimum_distance,
            map_idx=map_per_other,
        )
        self._last_raw_return_metrics = raw_return_metrics

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        partner_resampled = False
        if self.agent_sampling:
            self._episode_return.fill(0.0)
            self._reset_other_indices()
            partner_resampled = True
            self._update_minimum_distance()
        info = [{"agent_offsets": self.agent_offsets, "map_ids": self.map_ids, "num_envs": self.num_envs, "ego_indices": self.ego_indices}]
        if self.pbt_mode == "reactive":
            info[0]["other_indices"] = self.policy_per_slot
        if self.pbt_mode == "reactive":
            info[0]["map_policy_assignment_ids"] = self.map_policy_assignment_ids.copy()
        info[0]["partner_resampled"] = partner_resampled
        if self.agent_sampling and self._last_sampling_metrics:
            info[0]["sampling"] = {k: float(v) for k, v in self._last_sampling_metrics.items()}
        return self.observations, info

    def _allocate_replay(self, num_agents, map_ids):
        self.replay_actions = np.zeros((num_agents, self.resample_frequency, 1), dtype=np.int32)
        agent_ind = 0
        for _, map_id in enumerate(map_ids):
            num_rollout = self.other_actions.shape[0]
            sample_ind = np.random.randint(0, num_rollout)
            map_indices = np.where(self.actions_map_id[sample_ind] == map_id)[0][0]
            agent_offsets = self.actions_agent_offsets[sample_ind, map_indices:map_indices+2]
            num_agents_for_map = agent_offsets[1] - agent_offsets[0]
            if agent_ind + num_agents_for_map> num_agents:
                num_agents_for_map = num_agents - agent_ind
            self.replay_actions[agent_ind:agent_ind+num_agents_for_map] = self.other_actions[sample_ind, agent_offsets[0]:agent_offsets[0] + num_agents_for_map].copy()
            agent_ind += num_agents_for_map

    def step(self, actions):
        self.terminals[:] = 0
        self.actions[:] = actions
        if self.pbt_mode == "replay":
            self.actions[self.other_indices_arr] = self.replay_actions[self.other_indices_arr, self.tick, :]
            # self.actions[:] = self.replay_actions[:, self.tick, :]
        binding.vec_step(self.c_envs)
        if self.agent_sampling: # TODO: 현재는 Return 기반만 구현되어 있음
            self._update_minimum_distance()
            self._episode_return[self.ego_indices] += self.rewards[self.ego_indices]
        self.tick += 1
        info = []
        partner_resampled = False
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                aggregate, scenarios = split_aggregate_and_scenario(log)
                if scenarios and self.scenario_log_path:
                    append_scenario_logs(self.scenario_log_path, scenarios)
                if aggregate:
                    self._enrich_aggregate_metrics(aggregate)
                    info.append(aggregate)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            if self.agent_sampling:
                self._on_episode_end()
            partner_resampled = True
            self.tick = 0
            binding.vec_close(self.c_envs)
            agent_offsets, map_ids, num_envs, ego_indices = binding.shared(
                num_agents=self.num_agents,
                num_maps=self.num_maps,
                ego_ratio=self.ego_ratio,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                init_steps=self.init_steps,
                max_controlled_agents=self.max_controlled_agents,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_speed=self.goal_speed,
                aggressive_speed=self.aggressive_speed,
                map_dir=self.map_dir,
                sequential_map_sampling=False,  # Always use random sampling with replacement
            )
            self.agent_offsets = agent_offsets
            self.map_ids = map_ids
            self.num_envs = num_envs
            self.ego_indices = np.asarray(ego_indices, dtype=np.int64)
            ao = np.asarray(agent_offsets, dtype=np.int64)
            self.num_ego_per_env = [int(np.sum((self.ego_indices >= ao[i]) & (self.ego_indices < ao[i + 1]))) for i in range(num_envs)]
            self.other_mask = np.ones(self.num_agents, dtype=bool)
            self.other_mask[self.ego_indices] = False
            self.other_indices_arr = np.flatnonzero(self.other_mask).astype(np.int64)
            self._ego_index_set = set(self.ego_indices.tolist())
            if self.pbt_mode == "replay" and not self.agent_sampling: # TODO: UNIFORM SAMPLING은 옮겨줘야 함
                self._allocate_replay(self.num_agents, self.map_ids)

            env_ids = []
            seed = np.random.randint(0, 2**32 - 1)
            for i in range(num_envs):
                cur = agent_offsets[i]
                nxt = agent_offsets[i + 1]
                ego_local = (self.ego_indices - cur)[(self.ego_indices >= cur) & (self.ego_indices < nxt)].astype(np.int32).tolist()
                env_id = binding.env_init(
                    self.observations[cur:nxt],
                    self.actions[cur:nxt],
                    self.rewards[cur:nxt],
                    self.terminals[cur:nxt],
                    self.truncations[cur:nxt],
                    seed,
                    action_type=self._action_type_flag,
                    human_agent_idx=self.human_agent_idx,
                    reward_vehicle_collision=self.reward_vehicle_collision,
                    reward_offroad_collision=self.reward_offroad_collision,
                    reward_goal=self.reward_goal,
                    reward_goal_post_respawn=self.reward_goal_post_respawn,
                    aggressive_speed=self.aggressive_speed,
                    reward_head_diff=self.reward_head_diff,
                    reward_lane_dist=self.reward_lane_dist,
                    reward_speed=self.reward_speed,
                    goal_radius=self.goal_radius,
                    goal_behavior=self.goal_behavior,
                    goal_target_distance=self.goal_target_distance,
                    goal_speed=self.goal_speed,
                    collision_behavior=self.collision_behavior,
                    offroad_behavior=self.offroad_behavior,
                    dt=self.dt,
                    episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                    termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                    max_controlled_agents=self.max_controlled_agents,
                    map_id=map_ids[i],
                    max_agents=nxt - cur,
                    num_ego=self.num_ego_per_env[i],
                    ego_local_indices=ego_local,
                    ini_file="pufferlib/config/ocean/drive.ini",
                    init_steps=self.init_steps,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    map_dir=self.map_dir,
                    scenario_log_path=self.scenario_log_path or "",
                )
                env_ids.append(env_id)
            self.c_envs = binding.vectorize(*env_ids)

            binding.vec_reset(self.c_envs, seed)
            self.terminals[:] = 1
            if self.agent_sampling:
                self._reset_other_indices()
                self._update_minimum_distance()
        if len(info) == 0:
            info = [{"agent_offsets": self.agent_offsets, "map_ids": self.map_ids, "num_envs": self.num_envs, "ego_indices": self.ego_indices}]
        else:
            info[0]["agent_offsets"] = self.agent_offsets
            info[0]["map_ids"] = self.map_ids
            info[0]["num_envs"] = self.num_envs
            info[0]["ego_indices"] = self.ego_indices
        if self.pbt_mode == "reactive":
            info[0]["other_indices"] = self.policy_per_slot
        if self.pbt_mode == "reactive":
            info[0]["map_policy_assignment_ids"] = self.map_policy_assignment_ids.copy()
        info[0]["partner_resampled"] = partner_resampled
        if self.agent_sampling and self._last_sampling_metrics:
            info[0]["sampling"] = {k: float(v) for k, v in self._last_sampling_metrics.items()}

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
        )

        return states

    def get_global_partner_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'heading', 'id', 'speed', containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents
        num_partners = self.max_partner_objects
        states = {
            "x": np.zeros((num_agents, num_partners), dtype=np.float32),
            "y": np.zeros((num_agents, num_partners), dtype=np.float32),
            "heading": np.zeros((num_agents, num_partners), dtype=np.float32),
            "other_id": np.full((num_agents, num_partners), -1, dtype=np.int32),
            "ego_id": np.full((num_agents, ), -1, dtype=np.int32),
            "speed": np.zeros((num_agents, num_partners), dtype=np.float32),
        }

        binding.vec_get_global_partner_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["heading"],
            states["other_id"],
            states["ego_id"],
            states["speed"],
        )

        return states

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "is_vehicle": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["is_vehicle"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy
