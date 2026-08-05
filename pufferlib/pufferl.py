## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import glob
import ast
import json
import time
import random
import shutil
import subprocess
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.utils

try:
    from pufferlib import _C
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation"
    )

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter

rich.traceback.install(show_locals=False)

import signal  # Aggressively exit on ctrl+c

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None, other_policies=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config["seed"]
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        if config["use_pbt"]:
            total_agents = int(vecenv.num_agents_per_env * config["ego_ratio"]) * vecenv.num_environments
        self.total_agents = total_agents

        # Experience
        if config["batch_size"] == "auto" and config["bptt_horizon"] == "auto":
            raise pufferlib.APIUsageError("Must specify batch_size or bptt_horizon")
        elif config["batch_size"] == "auto":
            config["batch_size"] = total_agents * config["bptt_horizon"]
        elif config["bptt_horizon"] == "auto":
            config["bptt_horizon"] = config["batch_size"] // total_agents

        batch_size = config["batch_size"]
        horizon = config["bptt_horizon"]
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(f"Total agents {total_agents} <= segments {segments}")

        device = config["device"]
        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == "cuda" and config["cpu_offload"],
            device="cpu" if config["cpu_offload"] else device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *atn_space.shape,
            device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype],
        )
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents
        self.render = config["render"]
        self.render_interval = config["render_interval"]

        if self.render:
            ensure_drive_binary()

        # LSTM
        if config["use_rnn"]:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.num_agents_per_env = vecenv.num_agents_per_env
            self.lstm_h = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
            self.lstm_c = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
            if config["use_pbt"]:
                self.ego_ratio = config["ego_ratio"] # This should be divided with the segments & n
                num_ego = int(n * self.ego_ratio)
                self.num_ego_per_env = int(self.num_agents_per_env  * self.ego_ratio)
                self.num_other_per_env = int(self.num_agents_per_env  * (1 - self.ego_ratio))
                self.lstm_h = {i * self.num_agents_per_env: torch.zeros(num_ego, h, device=device) for i in range(total_agents // self.num_agents_per_env)}
                self.lstm_c = {i * self.num_agents_per_env: torch.zeros(num_ego, h, device=device) for i in range(total_agents // self.num_agents_per_env)}
                if config["pbt_mode"] == "reactive":
                    self.other_lstm_cs = []
                    self.other_lstm_hs = []
                    self.num_other_policies = len(other_policies)
                    self._other_lstm_hidden = h
                    for _ in range(self.num_other_policies):
                        self.other_lstm_hs.append({})
                        self.other_lstm_cs.append({})
        if config.get("use_pbt"):
            self._total_actions_buffer = np.zeros((n, 1), dtype=np.int64)

        # Minibatching & gradient accumulation
        minibatch_size = config["minibatch_size"]
        max_minibatch_size = config["max_minibatch_size"]
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f"minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly"
            )

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}")

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config["update_epochs"] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}"
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config["use_pbt"] and config["pbt_mode"] == "reactive":
            # Torch compile (other)
            self.uncompiled_other_policy = other_policies
            self.other_policies = []
            for other_policy in other_policies:
                self.other_policies.append(other_policy.eval())
        if config["compile"]:
            self.policy = torch.compile(policy, mode=config["compile_mode"])
            self.policy.forward_eval = torch.compile(policy, mode=config["compile_mode"])
            pufferlib.pytorch.sample_logits = torch.compile(
                pufferlib.pytorch.sample_logits, mode=config["compile_mode"]
            )
            if config["use_pbt"]:
                self.other_policies = [torch.compile(other_policy, mode=config["compile_mode"]) for other_policy in self.other_policies]
                self.other_policies_forward_eval = [torch.compile(other_policy, mode=config["compile_mode"]) for other_policy in self.other_policies]
        # Optimizer
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        elif config["optimizer"] == "muon":
            from heavyball import ForeachMuon

            warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
            import heavyball.utils

            heavyball.utils.compile_mode = config["compile_mode"] if config["compile"] else None
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config["total_timesteps"] // config["batch_size"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config["precision"]
        self.amp_context = contextlib.nullcontext()
        if config.get("amp", True) and config["device"] == "cuda":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(f"Invalid precision: {precision}: use float32 or bfloat16")

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def _sync_other_lstm(self, other_indices, key, device, force_reset=False):
        """Match other-policy LSTM buffers to env other_indices; reset on resample."""
        h = self._other_lstm_hidden
        for policy_idx, other_idx in enumerate(other_indices):
            batch_size = len(other_idx)
            hs = self.other_lstm_hs[policy_idx]
            cs = self.other_lstm_cs[policy_idx]
            if key not in hs or hs[key].shape[0] != batch_size:
                hs[key] = torch.zeros(batch_size, h, device=device)
                cs[key] = torch.zeros(batch_size, h, device=device)
            elif force_reset:
                hs[key].zero_()
                cs[key].zero_()

    def evaluate_pbt_replay(self):
        '''Collect rollout'''
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)
        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)
    
        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv() 
            ego_indices = []
            env_id = env_id * self.ego_ratio
            env_id = env_id.astype(np.int64)
            for i, info_i in enumerate(info):
                if "ego_indices" in info_i.keys():
                    ego = np.asarray(info_i["ego_indices"], dtype=np.int64)
                    offset = self.num_agents_per_env * i
                    ego_indices.extend((ego + offset))
            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            done_mask = d + t  # TODO: Handle truncations separately

            # ego_indices = self.ego_indices.reshape(-1)
            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)
            
            o_ego = o[ego_indices]
            o_ego_device = o_ego.to(device)
            r_ego = r[ego_indices]
            d_ego = d[ego_indices]
            mask_ego = mask[ego_indices]
            self.global_step += int(mask_ego.sum())
            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                ego_state = dict(
                    reward=r_ego,
                    done=d_ego,
                    env_id=env_id,
                    mask=mask_ego,
                )
                if config["use_rnn"]:
                    ego_state["lstm_h"] = self.lstm_h[env_id.start]
                    ego_state["lstm_c"] = self.lstm_c[env_id.start]
                logits_ego, value_ego = self.policy.forward_eval(o_ego_device, ego_state)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)

                r_ego = torch.clamp(r_ego, -1, 1)

            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = ego_state["lstm_h"]
                    self.lstm_c[env_id.start] = ego_state["lstm_c"]

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())
                ego_batch_rows = slice(batch_rows.start, batch_rows.stop)
                if config["cpu_offload"]:
                    self.observations[ego_batch_rows, l] = o_ego
                else:
                    self.observations[ego_batch_rows, l] = o_ego_device
                # stack transitions only ego
                self.actions[ego_batch_rows, l] = action_ego
                self.logprobs[ego_batch_rows, l] = logprob_ego
                self.rewards[ego_batch_rows, l] = r_ego
                self.terminals[ego_batch_rows, l] = d_ego.float()
                self.values[ego_batch_rows, l] = value_ego.flatten()
                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action_ego = action_ego.cpu().numpy()
                
                if isinstance(logits_ego, torch.distributions.Normal):
                    action_ego = np.clip(action_ego, self.vecenv.action_space.low, self.vecenv.action_space.high)
                if os.environ.get("PUFFER_BENCH_REPLAY"):
                    _t0 = time.perf_counter()
                total_actions = self._total_actions_buffer
                total_actions[ego_indices] = action_ego
                if os.environ.get("PUFFER_BENCH_REPLAY"):
                    _t1 = time.perf_counter()
                    if not hasattr(self, "_bench_total_actions"):
                        self._bench_total_actions = [0.0, 0]
                    self._bench_total_actions[0] += _t1 - _t0
                    self._bench_total_actions[1] += 1

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)
            profile("env", epoch)
            self.vecenv.send(total_actions)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        if os.environ.get("PUFFER_BENCH_REPLAY") and hasattr(self, "_bench_total_actions") and self._bench_total_actions[1] > 0:
            a, n = self._bench_total_actions
            print(f"[bench] pufferl total_actions: {a*1000:.3f}ms / {n} recvs = {a/n*1e6:.1f}us/recv")
        profile.end()
        return self.stats

    def evaluate_pbt(self):
        '''Collect rollout'''
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)
                for l in range(len(self.other_lstm_hs)):
                    for key in self.other_lstm_hs[l]:
                        self.other_lstm_hs[l][key].zero_()
                        self.other_lstm_cs[l][key].zero_()
    
        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()
            ego_indices = []
            other_indices = [[] for _ in range(self.num_other_policies)]
            partner_resampled = False
            env_id = env_id * self.ego_ratio
            env_id = env_id.astype(np.int64)
            for i, info_i in enumerate(info):
                if info_i.get("partner_resampled"):
                    partner_resampled = True
                if "ego_indices" in info_i.keys():
                    ego = np.asarray(info_i["ego_indices"], dtype=np.int64)
                    offset = self.num_agents_per_env * i
                    ego_indices.extend((ego + offset))
                    if self.num_other_per_env > 0:
                        for n in range(self.num_other_policies):
                            other = np.asarray(info_i["other_indices"][n], dtype=np.int64)
                            other_indices[n].extend((other + offset))

            # ego_indices = self.ego_indices.reshape(-1)
            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            done_mask = d + t  # TODO: Handle truncations separately

            if config["use_rnn"]:
                self._sync_other_lstm(other_indices, env_id.start, device, force_reset=partner_resampled)

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)
            o_ego = o[ego_indices]
            o_ego_device = o_ego.to(device)
            r_ego = r[ego_indices]
            d_ego = d[ego_indices]
            mask_ego = mask[ego_indices]
            o_others = []
            r_others = []
            d_others = []
            mask_others = []
            self.global_step += int(mask_ego.sum())
            for other_idx in other_indices:
                o_others.append(o[other_idx].to(device))
                r_others.append(r[other_idx])
                d_others.append(d[other_idx])
                mask_others.append(mask[other_idx])
                
            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                other_states = []
                ego_state = dict(
                    reward=r_ego,
                    done=d_ego,
                    env_id=env_id,
                    mask=mask_ego,
                )
                for i, other_mask in enumerate(mask_others):
                    other_state = dict(
                        reward=r_others[i],
                        done=d_others[i],
                        env_id=env_id,
                        mask=mask_others[i],
                    )
                    other_states.append(other_state)
                if config["use_rnn"]:
                    ego_state["lstm_h"] = self.lstm_h[env_id.start]
                    ego_state["lstm_c"] = self.lstm_c[env_id.start]
                    action_others = []
                    for i in range(len(other_states)):
                        other_state = other_states[i]
                        other_state["lstm_h"] = self.other_lstm_hs[i][env_id.start]
                        other_state["lstm_c"] = self.other_lstm_cs[i][env_id.start]
                        logits_other, _ = self.other_policies[i].forward_eval(o_others[i], other_state)
                        action_other, _, _ = pufferlib.pytorch.sample_logits(logits_other)
                        action_others.append(action_other)
                logits_ego, value_ego = self.policy.forward_eval(o_ego_device, ego_state)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)

                r_ego = torch.clamp(r_ego, -1, 1)
            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = ego_state["lstm_h"]
                    self.lstm_c[env_id.start] = ego_state["lstm_c"]
                    for i in range(len(action_others)):
                        self.other_lstm_hs[i][env_id.start] = other_states[i]["lstm_h"]
                        self.other_lstm_cs[i][env_id.start] = other_states[i]["lstm_c"]

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())
                ego_batch_rows = slice(batch_rows.start, batch_rows.stop)
                if config["cpu_offload"]:
                    self.observations[ego_batch_rows, l] = o_ego
                else:
                    self.observations[ego_batch_rows, l] = o_ego_device
                # stack transitions only ego
                self.actions[ego_batch_rows, l] = action_ego
                self.logprobs[ego_batch_rows, l] = logprob_ego
                self.rewards[ego_batch_rows, l] = r_ego
                self.terminals[ego_batch_rows, l] = d_ego.float()
                self.values[ego_batch_rows, l] = value_ego.flatten()
                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action_ego = action_ego.cpu().numpy()
                
                if isinstance(logits_ego, torch.distributions.Normal):
                    action_ego = np.clip(action_ego, self.vecenv.action_space.low, self.vecenv.action_space.high)
                total_actions = self._total_actions_buffer
                total_actions[ego_indices] = action_ego
                for i, other_idx in enumerate(other_indices):
                    action_other = action_others[i]
                    if isinstance(logits_other, torch.distributions.Normal):
                        action_other = np.clip(action_other, self.vecenv.action_space.low, self.vecenv.action_space.high)
                    total_actions[other_idx] = action_other.cpu().numpy()

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)
            profile("env", epoch)
            self.vecenv.send(total_actions)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t  # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)  # , non_blocking=True)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = state["lstm_h"]
                    self.lstm_c[env_id.start] = state["lstm_c"]

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            self.vecenv.send(action)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                adv,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss
            self.amp_context.__enter__()  # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

            if self.render and self.epoch % self.render_interval == 0:
                model_dir = os.path.join(self.config["data_dir"], f"{self.config['env']}_{self.logger.run_id}")
                model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

                if model_files:
                    # Take the latest checkpoint
                    latest_cpt = max(model_files, key=os.path.getctime)
                    bin_path = f"{model_dir}.bin"

                    # Export to .bin for rendering with raylib
                    try:
                        export_args = {"env_name": self.config["env"], "load_model_path": latest_cpt, **self.config}

                        export(
                            args=export_args,
                            env_name=self.config["env"],
                            vecenv=self.vecenv,
                            policy=self.uncompiled_policy,
                            path=bin_path,
                            silent=True,
                        )
                        pufferlib.utils.render_videos(
                            self.config, self.vecenv, self.logger, self.epoch, self.global_step, bin_path
                        )

                    except Exception as e:
                        print(f"Failed to export model weights: {e}")

        if (
            self.epoch > 1
            and self.config["eval"]["wosac_realism_eval"]
            and (((self.epoch - 1) % self.config["eval"]["eval_interval"] == 0) or done_training)
        ):
            config = self.config.copy()
            config["model_env_name"] = config["env"]
            config["env"] = "puffer_drive"
            config["ego_ratio"] = 1.0
            pufferlib.utils.run_wosac_eval_in_subprocess(config, self.logger, self.global_step)

        if (
            self.epoch > 1
            and self.config["eval"]["human_replay_eval"]
            and (((self.epoch - 1) % self.config["eval"]["eval_interval"] == 0) or done_training)
        ):
            config = self.config.copy()
            config["model_env_name"] = config["env"]
            config["env"] = "puffer_drive"
            pufferlib.utils.run_human_replay_eval_in_subprocess(config, self.logger, self.global_step)


    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        # Debug: estimate replay (other) success rate; should stay ~0.98 if replay is stable
        # score = (ego_score*ego_n + other_score*other_n)/n  =>  other_implicit = (score*n - ego_score*ego_n)/(n - ego_n)
        if config.get("use_pbt") and "score" in self.stats and "ego_score" in self.stats and "n" in self.stats and "ego_n" in self.stats:
            s, es, n, en = self.stats["score"], self.stats["ego_score"], self.stats["n"], self.stats["ego_n"]
            if n > en and en > 0:
                other_implicit = (s * n - es * en) / (n - en)
                self.stats["other_score_implicit"] = float(np.clip(other_implicit, 0, 1))
            if "policy_ego_n" in self.stats and "legacy_ego_n" in self.stats:
                pen = float(np.mean(self.stats["policy_ego_n"]))
                leg = float(np.mean(self.stats["legacy_ego_n"]))
                self.stats["policy_ego_n"] = pen
                self.stats["legacy_ego_n"] = leg
                if leg > 0.5:
                    self.stats["ego_metric_legacy_warning"] = leg

        device = config["device"]
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            "SPS": dist_sum(self.sps, device),
            "agent_steps": agent_steps,
            "uptime": time.time() - self.start_time,
            "epoch": int(dist_sum(self.epoch, device)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            # **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            # **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self.logger.log(logs, agent_steps)
                return logs
            else:
                return None
        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}.pt")
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": run_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0], c1="[cyan]", c2="[white]", b1="[bright_cyan]", b2="[bright_white]"):
        config = self.config
        sps = dist_sum(self.sps, config["device"])
        agent_steps = dist_sum(self.global_step, config["device"])
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True, show_header=False, border_style="bright_cyan")
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f"{b1}PufferLib {b2}3.0 {idx[0] * ' '}:blowfish:",
            f"{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%",
            f"{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%",
            f"{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%",
            f"{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%",
        )
        idx[0] = (idx[0] - 1) % 10

        s = Table(box=None, expand=True)
        remaining = "A hair past a freckle"
        if sps != 0:
            remaining = duration((config["total_timesteps"] - agent_steps) / sps, b2, c2)

        s.add_column(f"{c1}Summary", justify="left", vertical="top", width=10)
        s.add_column(f"{c1}Value", justify="right", vertical="top", width=14)
        s.add_row(f"{c2}Env", f"{b2}{config['env']}")
        s.add_row(f"{c2}Params", abbreviate(self.model_size, b2, c2))
        s.add_row(f"{c2}Steps", abbreviate(agent_steps, b2, c2))
        s.add_row(f"{c2}SPS", abbreviate(sps, b2, c2))
        s.add_row(f"{c2}Epoch", f"{b2}{self.epoch}")
        s.add_row(f"{c2}Uptime", duration(self.uptime, b2, c2))
        s.add_row(f"{c2}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf("Evaluate", b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf("  Env", c2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf("Train", b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf("  Learn", c2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.train_misc, b2, c2))

        l = Table(
            box=None,
            expand=True,
        )
        l.add_column(f"{c1}Losses", justify="left", width=16)
        l.add_column(f"{c1}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0
        dashboard_ignore_stats = {
            "partner_resampled",
            "metric_ego_n",
            "legacy_ego_n",
            "other_policy_n",
            "policy_ego_n",
            "ego_metric_legacy_warning",
        }

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            if metric in dashboard_ignore_stats:
                continue
            try:  # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            display_metric = metric[len("sampling/") :] if metric.startswith("sampling/") else metric
            u.add_row(f"{c2}{display_metric}", f"{b2}{value:.3f}")
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[0;0H" + capture.get())


def compute_puff_advantage(
    values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
):
    """CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version."""

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()
    torch.ops.pufferlib.compute_puff_advantage(
        values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
    )

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return f"{num / 1e3:.1f}K"
    elif num < 1e9:
        return f"{num / 1e6:.1f}M"
    elif num < 1e12:
        return f"{num / 1e9:.1f}B"
    else:
        return f"{num / 1e12:.2f}T"


def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"], b2, c2), f"{b2}{percent:2d}{c2}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()


class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        if epoch % self.frequency != 0:
            return

        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]["start"] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile["start"]
        profile["elapsed"] += delta
        profile["delta"] += delta

    def end(self):
        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof["delta"] > 0:
                prof["buffer"] = prof["delta"]
                prof["delta"] = 0


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent() / psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                    time.sleep(self.delay)
                    continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def downsample(arr, m):
    if len(arr) < m:
        return arr

    if m == 0:
        return [arr[-1]]

    orig_arr = arr
    last = arr[-1]
    arr = arr[:-1]
    arr = np.array(arr)
    n = len(arr)
    n = (n // m) * m
    arr = arr[-n:]
    downsampled = arr.reshape(m, -1).mean(axis=1)
    return np.concatenate([downsampled, [last]])


class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100 * time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass


class NeptuneLogger:
    def __init__(self, args, load_id=None, mode="async"):
        import neptune as nept

        neptune_name = args["neptune_name"]
        neptune_project = args["neptune_project"]
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def close(self, model_path):
        self.neptune["model"].track_files(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"


WANDB_IGNORE_ENV_KEYS = {
    "environment/num_envs",
    "environment/agent_offsets",
    "environment/map_ids",
    "environment/other_indices",
    "environment/ego_n",
    "environment/partner_resampled",
    "environment/policy_ego_n",
    "environment/metric_ego_n",
    "environment/legacy_ego_n",
    "environment/other_policy_n",
    "environment/ego_metric_legacy_warning",
}


class WandbLogger:
    def __init__(self, args, load_id=None, resume="allow"):
        import wandb

        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            name=args.get("wandb_name"),
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.wandb = wandb
        self.run_id = wandb.run.id

    def log(self, logs, step):
        ego_prefix = "environment/ego_"
        sampling_prefix = "environment/sampling/"
        logs_filtered = {}
        for k, v in logs.items():
            if k in WANDB_IGNORE_ENV_KEYS:
                continue
            if k.startswith(sampling_prefix):
                logs_filtered[f"sampling/{k[len(sampling_prefix):]}"] = v
            elif k.startswith(ego_prefix):
                logs_filtered[f"ego/{k[len(ego_prefix):]}"] = v
            else:
                logs_filtered[k] = v
        self.wandb.log(logs_filtered, step=step)

    def close(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print("World size", world_size)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args["neptune"]:
        logger = NeptuneLogger(args)
    elif args["wandb"]:
        logger = WandbLogger(args)

    train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}))
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20 * train_config["total_timesteps"]:
                all_logs.append(logs)

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    i = 0
    stats = {}
    while i < 32 or not stats:
        stats = pufferl.evaluate()
        i += 1

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs

def train_pbt(env_name, args=None, vecenv=None, policy=None, logger=None, config=None):
    args = args or load_config(env_name, config_dir=config)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print("World size", world_size)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    # if the pbt train mode is reactive, load other policy
    populations = [
        f for f in os.listdir(args["pbt"]["population_path"])
        if f.endswith(".pt")
    ]
    policies = None
    if args["pbt"]["pbt_mode"] == "reactive":
        policies = []
        for op in populations:
            args2 = args.copy()
            args2["load_model_path"] = os.path.join(args["pbt"]["population_path"], op)
            policy2 = load_policy(args2, vecenv, env_name)
            policies.append(policy2)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        for p, policy2 in enumerate(policies):
            policies[p] = policy2.to(local_rank)
            policies[p] = torch.nn.parallel.DistributedDataParallel(policies[p], device_ids=[local_rank], output_device=local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size
            for p, policy2 in enumerate(policies):
                policies[p].hidden_size = policy2.hidden_size
                policies[p].forward_eval = policy2.forward_eval
                policies[p] = policy2.to(local_rank)
        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)
    if args["neptune"]:
        logger = NeptuneLogger(args)
    elif args["wandb"]:
        logger = WandbLogger(args)

    train_config = dict(**args["train"], **args["pbt"],env=env_name, eval=args.get("eval", {}))
    pufferl = PuffeRL(train_config, vecenv, policy, logger, policies)

    all_logs = []
    while pufferl.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        if args["pbt"]["pbt_mode"] == "reactive":
            pufferl.evaluate_pbt()
        elif args["pbt"]["pbt_mode"] == "replay":
            pufferl.evaluate_pbt_replay()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()
        if logs is not None:
            if pufferl.global_step > 0.20 * train_config["total_timesteps"]:
                all_logs.append(logs)

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    i = 0
    stats = {}
    while i < 32 or not stats:
        if args["pbt"]["pbt_mode"] == "reactive":
            stats = pufferl.evaluate_pbt()
        elif args["pbt"]["pbt_mode"] == "replay":
            stats = pufferl.evaluate_pbt_replay()
        i += 1
    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs


def eval(env_name, args=None, vecenv=None, policy=None):
    """Evaluate a policy."""

    args = args or load_config(env_name)

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    human_replay_enabled = args["eval"]["human_replay_eval"]
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    args["env"]["num_maps"] = args["eval"]["wosac_num_maps"]
    args["env"]["sequential_map_sampling"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]

    if wosac_enabled:
        print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv" or not wosac_enabled, "WOSAC evaluation only supports PufferEnv backend."
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]
        args["base"]["env_name"] = "puffer_drive"
        env_name = "puffer_drive"
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        evaluator = WOSACEvaluator(args)

        # Collect ground truth trajectories from the dataset
        gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

        print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
        print(f"Number of controlled agents: {gt_trajectories['x'].shape[0]}")
        print(f"Number of evaluated agents: {np.sum(gt_trajectories['id'] >= 0)}")

        # Roll out trained policy in the simulator
        simulated_trajectories = evaluator.collect_simulated_trajectories(args, vecenv, policy)

        if args["eval"]["wosac_sanity_check"]:
            evaluator._quick_sanity_check(gt_trajectories, simulated_trajectories)

        # Analyze and compute metrics
        agent_state = vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
        results = evaluator.compute_metrics(
            gt_trajectories,
            simulated_trajectories,
            agent_state,
            road_edge_polylines,
            args["eval"]["wosac_aggregate_results"],
        )

        if args["eval"]["wosac_aggregate_results"]:
            import json

            print("\nWOSAC_METRICS_START")
            print(json.dumps(results))
            if args["eval"].get("wosac_save_results", True) and args.get("load_model_path"):
                id_ = args["load_model_path"]
                exp = id_.split("/")[-2]
                map_results = {id_[-11:-3]: results}
                save_result(f"/data/puffer/results/{exp}/wosac.json", map_results)
            print("WOSAC_METRICS_END")

        return results

    elif human_replay_enabled:
        print(f"Running human replay evaluation with {dataset_name} dataset.\n")
        from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

        backend = args["eval"].get("backend", "PufferEnv")
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["episode_length"] = 91  # WOMD scenario length
        args["env"]["termination_mode"] = 0 # Should be 0 for human replay evaluation
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        print(f"Effective number of scenarios used: {len(vecenv.driver_env.agent_offsets) - 1}")

        evaluator = HumanReplayEvaluator(args)

        # Run rollouts with human replays
        results = evaluator.rollout(args, vecenv, policy)

        import json

        print("HUMAN_REPLAY_METRICS_START")
        print(json.dumps(results))
        if args["eval"].get("human_replay_save_results", True) and args.get("load_model_path"):
            id_ = args["load_model_path"]
            exp = id_.split("/")[-2]
            map_results = {id_[-11:-3]: results}
            print(f"EXP {exp}")
            save_result(f"/data/puffer/results/{exp}/logreplay.json", map_results)
        print("HUMAN_REPLAY_METRICS_END")

        return results
    else:  # Standard evaluation: Render
        backend = args["vec"]["backend"]
        if backend != "PufferEnv":
            backend = "Serial"

        args["vec"] = dict(backend=backend, num_envs=1)
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        ob, info = vecenv.reset()
        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        device = args["train"]["device"]

        # Rebuild visualize binary if saving frames (for C-based rendering)
        if args["save_frames"] > 0:
            ensure_drive_binary()

        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        target_frames = int(args["save_frames"])
        if target_frames > 0:
            frames = []
            for _ in range(target_frames):
                render = driver.render()
                if render is not None:
                    frames.append(render)

                with torch.no_grad():
                    ob = torch.as_tensor(ob).to(device)
                    logits, value = policy.forward_eval(ob, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                    action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob = vecenv.step(action)[0]

            if not frames:
                raise pufferlib.APIUsageError(
                    "No render frames captured. Drive.render() returns None in raylib mode. "
                    "Use: python analyze/viz.py --model <ckpt.pt> --map-bin <map_XXX.bin> --out out.gif --use-xvfb"
                )

            import imageio

            imageio.mimsave(args["gif_path"], frames, fps=args["fps"], loop=0)
            print(f"Saved {args['gif_path']} ({len(frames)} frames)")
            return

        while True:
            render = driver.render()

            if driver.render_mode == "ansi" and render is not None:
                print("\033[0;0H" + render + "\n")
                time.sleep(1 / args["fps"])

            with torch.no_grad():
                ob = torch.as_tensor(ob).to(device)
                logits, value = policy.forward_eval(ob, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            ob = vecenv.step(action)[0]


def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb or neptune")

    method = args["sweep"].pop("method")
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f"Invalid sweep method {method}. See pufferlib.sweep")

    sweep = sweep_cls(args["sweep"])
    points_per_run = args["sweep"]["downsample"]
    target_key = f"environment/{args['sweep']['metric']}"
    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep.suggest(args)
        total_timesteps = args["train"]["total_timesteps"]
        all_logs = train(env_name, args=args)
        all_logs = [e for e in all_logs if target_key in e]
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = downsample([log["agent_steps"] for log in all_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            args["train"]["total_timesteps"] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args["train"]["total_timesteps"] = total_timesteps


def controlled_exp(env_name, args=None):
    """Run experiments with all combinations of specified parameter values."""
    import itertools
    from copy import deepcopy

    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Targeted experiments require either wandb or neptune")

    # Check if controlled_exp config exists
    if "controlled_exp" not in args:
        raise pufferlib.APIUsageError("No [controlled_exp.*] sections found in config")

    # Extract parameters from controlled_exp namespace
    params = {}
    for section, section_config in args["controlled_exp"].items():
        if isinstance(section_config, dict):
            for param, param_config in section_config.items():
                if isinstance(param_config, dict) and "values" in param_config:
                    params[f"{section}.{param}"] = param_config["values"]

    if not params:
        raise pufferlib.APIUsageError("No parameters with 'values' lists found in [controlled_exp.*] sections")

    # Generate all combinations
    keys = list(params.keys())
    combinations = list(itertools.product(*[params[k] for k in keys]))

    print(f"Running a total of {len(combinations)} experiments with parameters: {keys}")

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        exp_args = deepcopy(args)

        # Set parameters
        for key, value in zip(keys, combo):
            section, param = key.split(".")
            exp_args[section][param] = value

        print(f"\nExperiment {i}/{len(combinations)}: {dict(zip(keys, combo))}")

        # Train
        train(env_name, args=exp_args)

    print(f"\n✓ Completed all {len(combinations)} experiments")


def sanity(env_name, args=None):
    args = args or load_config(env_name)
    base_dir = Path(__file__).resolve().parent / "resources" / "drive" / "sanity"
    json_dir = base_dir / "sanity_jsons"
    binary_dir = base_dir / "sanity_binaries"

    available_maps = {p.stem: p for p in json_dir.glob("*.json")}
    selected = args.get("sanity_maps")
    if isinstance(selected, str):
        selected = [selected]

    if selected:
        missing = [name for name in selected if name not in available_maps]
        if missing:
            raise pufferlib.APIUsageError(f"Unknown sanity maps: {', '.join(sorted(missing))}")
        chosen = [(name, available_maps[name]) for name in selected]
    else:
        chosen = sorted(available_maps.items())

    if not chosen:
        raise pufferlib.APIUsageError(f"No sanity maps found in {json_dir}")

    from pufferlib.ocean.drive.drive import load_map

    binary_dir.mkdir(parents=True, exist_ok=True)
    binaries = []
    for idx, (name, json_path) in enumerate(chosen):
        output_path = binary_dir / f"{name}.bin"
        load_map(str(json_path), idx, str(output_path))
        binaries.append((name, output_path))

    runs = []
    for name, binary in binaries:
        map_zero = binary_dir / "map_000.bin"
        shutil.copy2(binary, map_zero)

        run_args = {
            **args,
            "env": {**args["env"], "num_maps": 1, "map_dir": str(binary_dir)},
            "train": {**args["train"], "render_map": str(map_zero)},
        }
        if run_args.get("wandb"):
            run_args["wandb_name"] = name

        print(f"Running sanity map '{name}' from {binary.name}")
        run_logs = train(env_name=env_name, args=run_args)
        runs.append({"map": name, "logs": run_logs})

    print("Sanity checklist:")
    for entry in runs:
        name = entry["map"]
        logs = entry.get("logs") or []
        final = logs[-1] if logs else {}
        score = final.get("environment/score")
        if score is None:
            status = "unknown (no score)"
        elif score >= 0.95:
            status = "✅ Solved"
        else:
            status = "❌ unsolved"
        print(f" - {name}: {status} (score={score})")

    return runs

def _curriculum_rollout_types(types_sorted, num_collect_rollout):
    """Build a staged curriculum type schedule of length num_collect_rollout.

    Split rollouts into len(types) stages. Stage s only uses types_sorted[:s+1]
    (easy→hard unlock), round-robin within the unlocked set.

    Example types=[0,1,2], M=9:
      [0, 0, 0,  0, 1, 0,  0, 1, 2]
    """
    types_sorted = list(types_sorted)
    m = int(num_collect_rollout)
    n_types = len(types_sorted)
    if m < 1:
        raise ValueError(f"num_collect_rollout must be >= 1, got {m}")
    if n_types < 1:
        raise ValueError("types_sorted must be non-empty")

    # Even stage sizes; remainder goes to later stages.
    base, rem = divmod(m, n_types)
    stage_sizes = [base + (1 if s >= n_types - rem else 0) for s in range(n_types)]
    rollout_types = []
    for s, size in enumerate(stage_sizes):
        allowed = types_sorted[: s + 1]
        for i in range(size):
            rollout_types.append(allowed[i % len(allowed)])
    assert len(rollout_types) == m
    return rollout_types


def _normalize_path_arg(value, name):
    """CLI paths go through ast.literal_eval; bare `...` becomes Ellipsis."""
    if value is None or value is Ellipsis:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if value in ("", "...", "Ellipsis", "None"):
        return ""
    return value


def _load_collect_order(collect_order_path, num_collect_rollout, num_checkpoints=0):
    """Load curriculum collect order JSON.

    Schema:
      {
        "type_to_population": {"0": "/path/to/pop0", "1": "/path/to/pop1"},
        "rollout_types": [0, 0, 1, ...]  # optional; auto curriculum schedule if omitted
      }

    When rollout_types is omitted, builds a staged schedule of length
    num_collect_rollout (early = easy only, later = mix including harder types).
    collect_num_checkpoints only limits how many *.pt are mixed per type.
    Returns (type_to_population, rollout_types, num_collect_rollout).
    """
    path = _normalize_path_arg(collect_order_path, "collect_order_path")
    if not path:
        raise ValueError(
            "collect_order_path is empty or invalid (did you pass ORDER_PATH=... from the docs? "
            "Use a real file, e.g. analyze/curriculum_collect_order.example.json)"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"collect_order_path not found: {path}")
    with open(path, "r") as f:
        order = json.load(f)

    type_to_population = order.get("type_to_population")
    if not isinstance(type_to_population, dict) or not type_to_population:
        raise ValueError("collect_order JSON must include non-empty type_to_population")

    type_to_population = {int(k): str(v) for k, v in type_to_population.items()}
    for t, pop in type_to_population.items():
        if not os.path.isdir(pop):
            raise FileNotFoundError(f"type_to_population[{t}] is not a directory: {pop}")

    types_sorted = sorted(type_to_population)
    rollout_types = order.get("rollout_types")
    if rollout_types is None or rollout_types == [] or rollout_types == "":
        rollout_types = _curriculum_rollout_types(types_sorted, num_collect_rollout)
    else:
        if not isinstance(rollout_types, list):
            raise ValueError("rollout_types must be a list when provided")
        if len(rollout_types) != int(num_collect_rollout):
            raise ValueError(
                f"len(rollout_types)={len(rollout_types)} != num_collect_rollout={num_collect_rollout}"
            )
        rollout_types = [int(t) for t in rollout_types]
        missing = sorted(set(rollout_types) - set(type_to_population))
        if missing:
            raise ValueError(
                f"rollout_types reference unknown types (not in type_to_population): {missing}"
            )

    # num_checkpoints is validated later when loading; keep total = requested M.
    _ = num_checkpoints
    return type_to_population, rollout_types, int(num_collect_rollout)


def _load_policies_from_population(population_path, args, vecenv, env_name, num_checkpoints=0):
    """Load *.pt checkpoints under population_path as eval policies.

    Checkpoints are sorted by filename. If num_checkpoints > 0, only the first N
    are loaded; 0 means load all. All loaded policies are mixed in each rollout.

    Returns (policies, checkpoint_filenames).
    """
    ckpts = sorted(f for f in os.listdir(population_path) if f.endswith(".pt"))
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoints under {population_path}")
    n = int(num_checkpoints)
    if n > 0:
        if n > len(ckpts):
            raise ValueError(
                f"collect_num_checkpoints={n} > available checkpoints ({len(ckpts)}) "
                f"under {population_path}: {ckpts}"
            )
        ckpts = ckpts[:n]
    print(f"  using {len(ckpts)} checkpoint(s) from {population_path}: {ckpts}")
    policies = []
    for op in ckpts:
        args2 = args.copy()
        args2["load_model_path"] = os.path.join(population_path, op)
        policy2 = load_policy(args2, vecenv, env_name)
        policies.append(policy2.eval())
    return policies, ckpts


def _build_population_key_table(type_to_ckpts):
    """Flatten (type -> checkpoint list) into global population keys.

    Returns:
      keys: list of {key, type, population, checkpoint}
      type_to_local_to_global: {type: np.ndarray[int32] local_idx -> global_key}
    """
    keys = []
    type_to_local_to_global = {}
    for t in sorted(type_to_ckpts):
        pop_path, ckpts = type_to_ckpts[t]
        lut = []
        for ckpt in ckpts:
            gid = len(keys)
            lut.append(gid)
            keys.append(
                {
                    "key": gid,
                    "type": int(t),
                    "population": str(pop_path),
                    "checkpoint": str(ckpt),
                }
            )
        type_to_local_to_global[int(t)] = np.asarray(lut, dtype=np.int32)
    return keys, type_to_local_to_global


def _write_population_manifest(population_path, keys, type_to_ckpts):
    """Write lookup table for saved/population_keys.npy (and shard copies)."""
    manifest = {
        "keys": keys,
        "type_to_checkpoints": {
            str(t): {"population": pop, "checkpoints": list(ckpts)}
            for t, (pop, ckpts) in sorted(type_to_ckpts.items())
        },
        "note": (
            "saved/population_keys.npy has shape (num_rollouts, num_agents). "
            "Entry [r, a] is an index into keys[]; keys[i] identifies which "
            "population checkpoint acted for that agent in that rollout."
        ),
    }
    out = os.path.join(population_path, "population_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out} ({len(keys)} population keys)")
    return out


def zero_shot(env_name, args=None, vecenv=None, policies=None):
    args = args or load_config(env_name)
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    args["env"]["num_maps"] = args["eval"]["wosac_num_maps"]
    args["env"]["sequential_map_sampling"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]
    print(f"Running zero-shot evaluation with {dataset_name} dataset.\n")
    from pufferlib.ocean.benchmark.evaluator import OtherReplayEvaluator

    backend = args["eval"].get("backend", "PufferEnv")
    args["vec"] = dict(backend=backend, num_envs=1)
    # args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
    args["env"]["episode_length"] = 91  # WOMD scenario length
    args2 = args.copy()

    if (
        args["pbt"]["pbt_mode"] == "save-population"
        and not args.get("load_multiple_model_path")
    ):
        # Population trajectory collect. Pairwise zero-shot eval always passes
        # --load-multiple-model-path and must not enter this branch even though
        # drive.ini defaults pbt_mode=save-population.
        num_collect_rollout = int(args["pbt"].get("num_collect_rollout", 50))
        start_idx = int(args["pbt"].get("collect_start_idx", 0))
        end_idx = int(args["pbt"].get("collect_end_idx", num_collect_rollout))
        collect_order_path = _normalize_path_arg(
            args["pbt"].get("collect_order_path"), "collect_order_path"
        )
        collect_num_checkpoints = int(args["pbt"].get("collect_num_checkpoints", 0) or 0)
        skip_smoke = bool(args["pbt"].get("skip_collect_smoke_test", True))

        # Validate order / paths before the expensive 10k-map env load.
        policies_by_type = None
        rollout_types = None
        type_to_population = None
        if collect_order_path:
            type_to_population, rollout_types, num_collect_rollout = _load_collect_order(
                collect_order_path,
                num_collect_rollout,
                num_checkpoints=collect_num_checkpoints,
            )
            if end_idx > num_collect_rollout:
                raise ValueError(
                    f"collect_end_idx={end_idx} > num_collect_rollout={num_collect_rollout}"
                )

        if not (0 <= start_idx < end_idx <= num_collect_rollout):
            raise ValueError(
                "Need 0 <= collect_start_idx < collect_end_idx <= num_collect_rollout; "
                f"got start_idx={start_idx}, end_idx={end_idx}, num_collect_rollout={num_collect_rollout}"
            )

        # Keep 910-step collect horizon, but disable Drive.step mid-episode map rebuild
        # (which switches to sequential_map_sampling=False and breaks env reuse).
        # store collect_horizon outside env kwargs — Drive.__init__ does not accept it.
        collect_horizon = int(args["env"].get("resample_frequency", 910) or 910)
        args["env"]["resample_frequency"] = 0
        args["env"]["num_maps"] = 10000
        args["env"]["sequential_map_sampling"] = True
        # Used by OtherReplayEvaluator.collect_rollouts (not passed into Drive).
        args["collect_horizon"] = collect_horizon

        print(
            f"Collect setup: maps={args['env']['num_maps']}, "
            f"collect_horizon={collect_horizon}, resample_frequency=0 (env reuse), "
            f"skip_smoke={skip_smoke}, shard=[{start_idx},{end_idx})/{num_collect_rollout}"
        )
        vecenv = load_env(env_name, args)
        evaluator = OtherReplayEvaluator(args)

        if collect_order_path:
            policies_by_type = {}
            ckpts_by_type = {}
            for t, pop_path in type_to_population.items():
                print(f"Loading policies for type={t} from {pop_path}")
                policies_by_type[t], ckpts_by_type[t] = _load_policies_from_population(
                    pop_path,
                    args,
                    vecenv,
                    env_name,
                    num_checkpoints=collect_num_checkpoints,
                )
            type_to_ckpts = {
                t: (type_to_population[t], ckpts_by_type[t]) for t in type_to_population
            }
            from collections import Counter
            counts = Counter(rollout_types)
            print(
                f"Curriculum collect order: {collect_order_path} "
                f"(out={args['pbt']['population_path']}, "
                f"collect_num_checkpoints={collect_num_checkpoints or 'all'}, "
                f"num_collect_rollout={num_collect_rollout}, "
                f"type_counts={dict(sorted(counts.items()))})"
            )
            print(f"  rollout_types={rollout_types}")
        else:
            policies, ckpts = _load_policies_from_population(
                args["pbt"]["population_path"],
                args,
                vecenv,
                env_name,
                num_checkpoints=collect_num_checkpoints,
            )
            type_to_ckpts = {
                -1: (args["pbt"]["population_path"], ckpts),
            }

        pop_keys_table, type_to_local_to_global = _build_population_key_table(type_to_ckpts)
        _write_population_manifest(args["pbt"]["population_path"], pop_keys_table, type_to_ckpts)

        shard_len = end_idx - start_idx
        split_dir = os.path.join(args["pbt"]["population_path"], "splits")
        os.makedirs(split_dir, exist_ok=True)
        tag = f"{start_idx:06d}_{end_idx:06d}"
        fp_a = os.path.join(split_dir, f"actions_{tag}.npy")
        fp_ao = os.path.join(split_dir, f"agent_offsets_{tag}.npy")
        fp_m = os.path.join(split_dir, f"map_ids_{tag}.npy")
        fp_gid = os.path.join(split_dir, f"global_ids_{tag}.npy")
        fp_types = os.path.join(split_dir, f"types_{tag}.npy")
        fp_pop = os.path.join(split_dir, f"population_keys_{tag}.npy")
        mm_a = mm_ao = mm_m = mm_gid = mm_types = mm_pop = None
        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            if policies_by_type is not None:
                rollout_type = int(rollout_types[global_i])
                policies = policies_by_type[rollout_type]
            else:
                rollout_type = -1
            other_action_buf, agent_offsets, map_ids, global_ids, local_policy_ids = (
                evaluator.collect_rollouts(args, vecenv, policies)
            )
            lut = type_to_local_to_global[rollout_type]
            if int(local_policy_ids.max()) >= int(lut.shape[0]):
                raise ValueError(
                    f"Rollout {global_i}: local policy id {int(local_policy_ids.max())} "
                    f">= num policies {int(lut.shape[0])} for type={rollout_type}"
                )
            population_keys = lut[local_policy_ids]
            agent_offsets = np.asarray(agent_offsets, dtype=np.int32, order="C")
            map_ids = np.asarray(map_ids, dtype=np.int32, order="C")
            global_ids = np.asarray(global_ids, dtype=np.int32, order="C")
            population_keys = np.asarray(population_keys, dtype=np.int32, order="C")
            if local_i == 0:
                na, T, c = other_action_buf.shape
                jo = int(agent_offsets.size)
                km = int(map_ids.size)
                nm, me = int(global_ids.shape[0]), int(global_ids.shape[1])
                mm_a = np.lib.format.open_memmap(
                    fp_a,
                    mode="w+",
                    dtype=other_action_buf.dtype,
                    shape=(shard_len, na, T, c),
                )
                mm_ao = np.lib.format.open_memmap(
                    fp_ao, mode="w+", dtype=np.int32, shape=(shard_len, jo)
                )
                mm_m = np.lib.format.open_memmap(
                    fp_m, mode="w+", dtype=np.int32, shape=(shard_len, km)
                )
                mm_gid = np.lib.format.open_memmap(
                    fp_gid, mode="w+", dtype=np.int32, shape=(shard_len, nm, me)
                )
                mm_types = np.lib.format.open_memmap(
                    fp_types, mode="w+", dtype=np.int32, shape=(shard_len,)
                )
                mm_pop = np.lib.format.open_memmap(
                    fp_pop, mode="w+", dtype=np.int32, shape=(shard_len, na)
                )
            if other_action_buf.shape != (na, T, c):
                raise ValueError(
                    f"Rollout {global_i} action shape {other_action_buf.shape} != "
                    f"first-rollout shape {(na, T, c)}; env layout changed unexpectedly"
                )
            if population_keys.shape != (na,):
                raise ValueError(
                    f"Rollout {global_i} population_keys shape {population_keys.shape} != ({na},)"
                )
            mm_a[local_i] = np.ascontiguousarray(other_action_buf)
            mm_ao[local_i] = agent_offsets.reshape(mm_ao.shape[1:])
            mm_m[local_i] = map_ids.reshape(mm_m.shape[1:])
            mm_gid[local_i] = global_ids
            mm_types[local_i] = rollout_type
            mm_pop[local_i] = population_keys
            # Env stays alive: collect_rollouts() calls reset(); resample_frequency=0
            # prevents Drive.step from rebuilding maps mid-horizon.
            type_msg = f", type={rollout_type}" if policies_by_type is not None else ""
            print(
                f"Collected shard {local_i + 1}/{shard_len} "
                f"(global rollout {global_i + 1}/{num_collect_rollout}{type_msg}), "
                f"shape: {other_action_buf.shape}, "
                f"pop_keys={np.unique(population_keys).tolist()}"
            )
        del mm_a, mm_ao, mm_m, mm_gid, mm_types, mm_pop
        print(f"Wrote {fp_types}")
        print(f"Wrote {fp_pop}")
        print(
            f"Wrote split memmaps under {split_dir} tag={tag} "
            f"(shard_len={shard_len}, agents={na}, T={T}). "
            f"Merge with: python analyze/data_concat.py --population-path {args['pbt']['population_path']} "
            f"--total-rollouts {num_collect_rollout}"
        )
        if skip_smoke:
            try:
                vecenv.close()
            except Exception:
                pass
            print("Skipped replay smoke-test (--pbt.skip-collect-smoke-test True).")
        elif start_idx == 0 and end_idx == num_collect_rollout:
            actions_rd = np.load(fp_a, mmap_mode="r")
            for i in range(shard_len):
                evaluator.replay_rollouts(args, vecenv, np.asarray(actions_rd[i], dtype=actions_rd.dtype))
                # replay may advance tick; reset between smoke iters
            vecenv.close()
            print("Finished replay smoke-test (full single-shard run).")
        else:
            try:
                vecenv.close()
            except Exception:
                pass
            print(
                "Skipped replay smoke-test for partial shard; run data_concat.py then replay training/eval."
            )
        return None

    args["env"]["num_maps"] = 10000
    vecenv = vecenv or load_env(env_name, args)

    args["load_model_path"] = args["load_multiple_model_path"][0]
    policy1 = load_policy(args, vecenv, env_name)

    args2["load_model_path"] = args["load_multiple_model_path"][1]
    policy2 = load_policy(args2, vecenv, env_name)

    print(f"Effective number of scenarios used: {len(vecenv.driver_env.agent_offsets) - 1}")
    parts_population = args["load_multiple_model_path"][1].split("/") 
    parts_ego = args["load_multiple_model_path"][0].split("/")
    evaluator = OtherReplayEvaluator(args, mode=parts_population[4], exp=parts_ego[4])

    # Run save replay
    # todo: randomly save the state for calculating log diff

    if args["zero_shot_mode"] == "save-replay":
        results = evaluator.save_replay(args, vecenv, policy1, policy2)
    elif args["zero_shot_mode"] == "replay":
        results = evaluator.play_replay(args, vecenv, policy1, policy2)
    elif args["zero_shot_mode"] == "reactive-play":
        results = evaluator.play_reactive(args, vecenv, policy1, policy2)

    return results

def linear_probe(env_name, args=None, vecenv=None, policy=None):
    from pufferlib.ocean.benchmark.linear_probe import LinearProbe
    args = args or load_config(env_name)
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    # args["env"]["num_maps"] = args["eval"]["wosac_num_maps"]
    args["env"]["num_maps"] = 300
    args["env"]["sequential_map_sampling"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]
    print(f"Running linear_probing with {dataset_name} dataset.\n")
    from pufferlib.ocean.benchmark.evaluator import OtherReplayEvaluator

    backend = args["eval"].get("backend", "PufferEnv")
    args["vec"] = dict(backend=backend, num_envs=1)
    # args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
    args["env"]["episode_length"] = 91  # WOMD scenario length

    vecenv = vecenv or load_env(env_name, args)
    args2 = args.copy()
    args["load_model_path"] = args["load_multiple_model_path"][0]
    policy1 = load_policy(args, vecenv, env_name)
    if "generate_" in args["lp_mode"]:
        args2["load_model_path"] = args["load_multiple_model_path"][1]
        policy2 = load_policy(args2, vecenv, env_name)
    else:
        policy2 = None
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)
    lp_module = LinearProbe(args)
    if "generate" in args["lp_mode"]:
        lp_module.make_dataset(args, vecenv, policy1, policy2)
    elif args["lp_mode"] == "train":
        lp_module.train(args, policy1, 10)
        lp_module.train(args, policy1, 20)
        lp_module.train(args, policy1, 30)
        lp_module.train(args, policy1, 40)
    elif args["lp_mode"] == "evaluate":
        other_model_id = args["load_multiple_model_path"][1][-11:-3]
        lp_module.evaluate(args, policy1, other_model_id, 10)
        lp_module.evaluate(args, policy1, other_model_id, 20)
        lp_module.evaluate(args, policy1, other_model_id, 30)
        lp_module.evaluate(args, policy1, other_model_id, 40)

        lp_module.evaluate(args, policy1, other_model_id, 10, mode="replay")
        lp_module.evaluate(args, policy1, other_model_id, 20, mode="replay")
        lp_module.evaluate(args, policy1, other_model_id, 30, mode="replay")
        lp_module.evaluate(args, policy1, other_model_id, 40, mode="replay")

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args["train"], env=args["env_name"], tag=args["tag"])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args["neptune"], wandb=args["wandb"])

    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

def save_result(path, res):
        import json
        import os
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = [data]
        data.append(res)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def export(args=None, env_name=None, vecenv=None, policy=None, path=None, silent=False):
    args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        if not silent:
            print(name, param.shape, param.data.cpu().numpy().ravel()[0])

    weights = np.concatenate(weights)
    if path is None:
        path = f"pufferlib/resources/drive/{args['env_name']}_weights.bin"

    weights.tofile(path)

    if not silent:
        print(f"Saved {len(weights)} weights to {path}")


def ensure_drive_binary():
    """Delete existing visualize binary and rebuild it. This ensures the
    binary is always up-to-date with the latest code changes.
    """
    if os.path.exists("./visualize"):
        os.remove("./visualize")

    try:
        result = subprocess.run(
            ["bash", "scripts/build_ocean.sh", "visualize", "local"], capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            raise RuntimeError("Failed to build visualize binary for rendering")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Build timed out")
    except Exception as e:
        raise RuntimeError(f"Build error: {e}")


def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    env_name = args["env_name"]
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args["train"]["env_batch_size"])


def load_env(env_name, args):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    if env_name == "puffer_drive":
        env_kwargs = {**args["env"]}
    elif env_name == "puffer_drive_pbt":
        env_kwargs = {**args["env"], **args.get("pbt", {})}
    else:
        return pufferlib.vector.make(make_env, env_kwargs={**args["env"]}, **args["vec"])

    sp = args.get("eval", {}).get("scenario_log_path")
    if sp is not None:
        env_kwargs["scenario_log_path"] = sp
    return pufferlib.vector.make(make_env, env_kwargs=env_kwargs, **args["vec"])


def load_policy(args, vecenv, env_name=""):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)

    device = args["train"]["device"]
    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(vecenv.driver_env, **args["policy"])

    rnn_name = args["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args["rnn_name"])
        policy = rnn_cls(vecenv.driver_env, policy, **args["rnn"])

    policy = policy.to(device)

    load_id = args["load_id"]
    if load_id is not None:
        if args["neptune"]:
            path = NeptuneLogger(args, load_id, mode="read-only").download()
        elif args["wandb"]:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError("No run id provided for eval")

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args["load_model_path"]
    if load_path == "latest":
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        # state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        # optim_state = torch.load(state_path)['optimizer_state_dict']
        # pufferl.optimizer.load_state_dict(optim_state)

    return policy


def load_config(env_name, config_dir=None):
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to a pretrained checkpoint")
    # for zero-shot evaluation
    parser.add_argument("--load-multiple-model-path", type=str, default=[], nargs="+", help="Path to a pretrained multiple checkpoints")
    parser.add_argument('--zero-shot-mode', type=str, default='save-replay',    
        choices=['reactive-play', 'replay', 'save-replay']
    )
    parser.add_argument('--lp-mode', type=str, default='generate', 
        choices=['generate', 'train', 'evaluate', 'generate_replay', 'generate_reactive']
    )
    parser.add_argument(
        "--load-id", type=str, default=None, help="Kickstart/eval from from a finished Wandb/Neptune run"
    )
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200, help="Max number of sweep runs")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument("--sanity-maps", nargs="*", default=None, help="Optional list of sanity map base names to run")
    args = parser.parse_known_args()[0]

    if config_dir is None:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        print("Using custom config dir:", config_dir)
        puffer_dir = config_dir

    # Load defaults and config
    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")
    if env_name == "default":
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p["base"]["env_name"].split():
                break
        else:
            raise pufferlib.APIUsageError("No config for env_name {}".format(env_name))

    # Dynamic help menu from config
    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    for section in p.sections():
        for key in p[section]:
            fmt = f"--{key}" if section == "base" else f"--{section}.{key}"
            parser.add_argument(fmt.replace("_", "-"), default=puffer_type(p[section][key]), type=puffer_type)

    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args["train"]["use_rnn"] = args["rnn_name"] is not None
    return args


def main():
    err = "Usage: puffer [train, eval, sweep, controlled_exp, autotune, profile, export, sanity] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    if mode == "train_pbt":
        config_dir = "pufferlib/ocean/drive_pbt"
        train_pbt(env_name=env_name, config=config_dir)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "zeroshot":
        zero_shot(env_name=env_name)
    elif mode == "linear_probe":
        linear_probe(env_name=env_name)
    elif mode == "controlled_exp":
        controlled_exp(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    elif mode == "sanity":
        sanity(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    main()
