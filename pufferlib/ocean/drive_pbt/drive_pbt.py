import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pufferlib.pufferl import load_policy

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
        pbt_mode="reactive", # for pbt
        population_path=None, # for replay
        ego_ratio=0.0, # for replay
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
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
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
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
        super().__init__(buf=buf)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
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
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        self.ego_ratio = ego_ratio
        self.population_path = population_path
        self.pbt_mode = pbt_mode
        self._allocate_ego_indices(self.num_agents)
        if pbt_mode == "replay":
            # Load Replay
            npz = np.load(os.path.join(self.population_path, "replay", "other_actions_int16.npz"), allow_pickle=True)
            self.other_actions = npz['actions']
            self.actions_agent_offsets = npz['agent_offsets']
            self.actions_map_id = npz['map_ids']
            del npz
            self._allocate_replay(self.num_agents, self.map_ids)
        else:
            populations = [
                f for f in os.listdir(population_path)
                if f.endswith(".pt")
            ]
            self.num_other_policies = len(populations)
            self._allocate_other_indices(self.num_agents)

    def reset(self, seed=0):    
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        info = [{"agent_offsets": self.agent_offsets, "map_ids": self.map_ids, "num_envs": self.num_envs, "ego_indices": self.ego_indices}]
        if self.pbt_mode == "reactive":
            info[0]["other_indices"] = self.other_indices
        return self.observations, info

    def _allocate_ego_indices(self, num_agents, agent_offsets=None):
        num_ego = int(num_agents * self.ego_ratio)
        num_ego = max(0, min(num_ego, num_agents))
        required = np.empty((0,), dtype=np.int64)
        if agent_offsets is not None and num_ego > 0:
            agent_offsets = np.asarray(agent_offsets, dtype=np.int64)
            assert agent_offsets.ndim == 1 and agent_offsets.size >= 2
            assert agent_offsets[0] == 0
            assert agent_offsets[-1] <= num_agents 

            # Ensure at least 1 ego per map
            candidates = []
            for i in range(agent_offsets.size - 1):
                s, e = int(agent_offsets[i]), int(agent_offsets[i + 1])
                if e > s:
                    candidates.append(np.random.randint(s, e))

            if len(candidates) > 0:
                candidates = np.asarray(candidates, dtype=np.int64)

                if candidates.size > num_agents:
                    pick = np.random.choice(candidates.size, size=num_agents, replace=False)
                    required = candidates[pick]
                else:
                    required = candidates

                num_ego = max(num_ego, required.size)
                num_ego = min(num_ego, num_agents)

        if num_ego == 0:
            self.ego_indices = np.empty((0,), dtype=np.int64)
            self.other_mask = np.ones(num_agents, dtype=bool)
            return

        all_idx = np.arange(num_agents, dtype=np.int64)
        remaining = np.setdiff1d(all_idx, required, assume_unique=False)

        extra_n = num_ego - required.size
        extra = (
            np.random.choice(remaining, size=extra_n, replace=False).astype(np.int64, copy=False)
            if extra_n > 0 else np.empty((0,), dtype=np.int64)
        )
        self.ego_indices = np.concatenate([required, extra])
        np.random.shuffle(self.ego_indices)
        self.other_mask = np.ones(num_agents, dtype=bool)
        self.other_mask[self.ego_indices] = False

    def _allocate_other_indices(self, num_agents):
        other_indices = np.setdiff1d(np.arange(num_agents), self.ego_indices, assume_unique=False)
        np.random.shuffle(other_indices)
        splits = np.array_split(other_indices, self.num_other_policies)
        self.other_indices = [s.astype(np.int64, copy=False) for s in splits]

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
        self.actions = actions
        if self.pbt_mode == "replay":
            # allocate replay actions
            replay_actions_t = self.replay_actions[:, self.tick, :]
            self.actions[self.other_mask] = replay_actions_t[self.other_mask]
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            binding.vec_close(self.c_envs)
            agent_offsets, map_ids, num_envs = binding.shared(
                num_agents=self.num_agents,
                num_maps=self.num_maps,
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
            self._allocate_ego_indices(self.num_agents)
            if self.pbt_mode == "replay":
                self._allocate_replay(self.num_agents, self.map_ids)
            else:
                self._allocate_other_indices(self.num_agents)
            env_ids = []
            seed = np.random.randint(0, 2**32 - 1)
            for i in range(num_envs):
                cur = agent_offsets[i]
                nxt = agent_offsets[i + 1]
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
                    max_controlled_agents=self.max_controlled_agents,
                    map_id=map_ids[i],
                    max_agents=nxt - cur,
                    ini_file="pufferlib/config/ocean/drive.ini",
                    init_steps=self.init_steps,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    map_dir=self.map_dir,
                )
                env_ids.append(env_id)
            self.c_envs = binding.vectorize(*env_ids)

            binding.vec_reset(self.c_envs, seed)
            self.terminals[:] = 1
        if len(info) == 0:
            info = [{"agent_offsets": self.agent_offsets, "map_ids": self.map_ids, "num_envs": self.num_envs, "ego_indices": self.ego_indices}]
        else:
            info[0]["agent_offsets"] = self.agent_offsets
            info[0]["map_ids"] = self.map_ids
            info[0]["num_envs"] = self.num_envs
            info[0]["ego_indices"] = self.ego_indices
        if self.pbt_mode == "reactive":
            info[0]["other_indices"] = self.other_indices
        # print(f"Rewards {self.rewards.max()} {self.rewards.mean()}")
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
