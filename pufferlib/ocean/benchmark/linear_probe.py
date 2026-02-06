"""WOSAC evaluation class for PufferDrive."""

import torch
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, functools

import torch.nn as nn
import torch 
import pufferlib.pytorch
import pufferlib.spaces
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from collections import OrderedDict

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, obs, labels, actions=None):
        self.obs = obs
        self.labels = labels
        # actions are not determined to use

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        return self.obs[index], self.labels[index]

class Probe(nn.Module):
    def __init__(self, hidden_size=128, future_step=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.classifier =nn.Linear(hidden_size, 64)
        self.future_step = future_step

    def forward(self, context):
        logits = self.classifier(context)
        return logits
    
    def forward_eval(self, context):
        logits = self.forward(context)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        return probs, pred_class
    
    def loss(self, pred_logits, expert_labels):
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_logits, expert_labels)
        
        # compute accuracy
        pred_class = torch.argmax(pred_logits, dim=-1)
        correct = (pred_class == expert_labels).sum().item()
        total = expert_labels.numel()
        accuracy = correct / total
        return loss, accuracy, pred_class
    
class LinearProbe:
    """Evaluates policies against other policies replays in PufferDrive."""

    def __init__(self, config: Dict):
        self.config = config
        self.sim_steps = 91
        self.num_epoch = 15
        if "reactive" in config["lp_mode"]:
            self.mode = "reactive"
        elif "replay" in config["lp_mode"]:
            self.mode = "replay"
        else:
            self.mode = None

    def save_result(self, path, res):
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

    def get_dataloader(self, data_path):
        with np.load(os.path.join(data_path)) as npz:
            partner_obs = npz["partner_obs"]
            labels = npz["label"]
        dataset = ProbeDataset(partner_obs, labels)
        n = len(dataset)
        n_eval = int(round(n * 0.1))
        n_train = n - n_eval
        train_ds = Subset(dataset, range(0, n_train))
        eval_ds  = Subset(dataset, range(n_train, n))
        del dataset
        train_loader = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,     
        num_workers=16,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=16,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader, eval_loader
    
    def train(self, args, policy, future_timestep):
        model = Probe(hidden_size=64, future_step=future_timestep).cuda() # TODO: args fixed num
        model_id = args["load_model_path"][-11:-3]
        data_path = f"linear_prob/{model_id}/future_{future_timestep}.npz"
        train_loader, eval_loader = self.get_dataloader(data_path)
        optim = AdamW(model.parameters(), lr=0.001, eps=0.0005)
        best_loss = 1e6
        policy.eval()
        for ep in tqdm(range(self.num_epoch)):
            model.train()
            loss_sum = 0.0
            acc_sum  = 0.0
            for k, batch in enumerate(train_loader):
                obs, labels = batch
                obs = obs.to("cuda")
                labels = labels.to("cuda")
                with torch.no_grad():
                    ob_tensor = torch.as_tensor(obs).to("cuda")
                    partner_feat = policy.policy.partner_encoder(ob_tensor)
                pred_pos = model.forward(partner_feat)
                pred_loss, acc, cls = model.loss(pred_pos, labels)
                optim.zero_grad()
                pred_loss.backward()
                optim.step()
                loss_val = float(pred_loss.detach().item())
                acc_val = float(acc.detach().item()) if torch.is_tensor(acc) else float(acc)

                loss_sum += loss_val
                acc_sum += acc_val
            if ep % 2 == 0:
                model.eval()
                eval_loss_sum = 0.0
                eval_acc_sum  = 0.0
                for j, batch in enumerate(eval_loader):
                    obs, labels = batch
                    obs = obs.to("cuda")
                    labels = labels.to("cuda")
                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(obs).to("cuda")
                        partner_feat = policy.policy.partner_encoder(ob_tensor)
                        pred_pos = model.forward(partner_feat)
                        pred_loss, acc, cls = model.loss(pred_pos, labels)
                        eval_loss_val = float(pred_loss.detach().item())
                        eval_acc_val = float(acc.detach().item()) if torch.is_tensor(acc) else float(acc)
                        eval_loss_sum += eval_loss_val
                        eval_acc_sum += eval_acc_val
                eval_loss_mean = eval_loss_sum /(j+1) 

                if eval_loss_mean < best_loss:
                    best_loss = eval_loss_mean
                    save_dir = f"linear_prob/{model_id}/model_{future_timestep}.pt"
                    torch.save(model, save_dir)
                    print(f'EPOCH {ep} gets BEST!')
                print(f"[Epoch {ep}] loss={eval_loss_mean:.4f}, acc={eval_acc_sum/(j+1):.4f}")

    def evaluate(self, args, policy, other_model_id, future_timestep, mode="reactive"):
        model_id = args["load_model_path"][-11:-3]
        model = torch.load(
            f"linear_prob/{model_id}/model_{future_timestep}.pt",
            weights_only=False,
            map_location="cpu",
        ).cuda()
        model.eval()
        data_path = f"linear_prob/{model_id}/{other_model_id}/{mode}/future_{future_timestep}.npz"
        with np.load(os.path.join(data_path)) as npz:
            partner_obs = npz["partner_obs"]
            labels = npz["label"]
        dataset = ProbeDataset(partner_obs, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=16,
            prefetch_factor=4,
            pin_memory=True
        )
        eval_loss_sum = 0
        eval_acc_sum = 0
        for j, batch in enumerate(dataloader):
            obs, labels = batch
            obs = obs.to("cuda")
            labels = labels.to("cuda")
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to("cuda")
                partner_feat = policy.policy.partner_encoder(ob_tensor)
                pred_pos = model.forward(partner_feat)
                pred_loss, acc, cls = model.loss(pred_pos, labels)
                eval_loss_val = float(pred_loss.detach().item())
                eval_acc_val = float(acc.detach().item()) if torch.is_tensor(acc) else float(acc)
                eval_loss_sum += eval_loss_val
                eval_acc_sum += eval_acc_val
        eval_loss_mean = eval_loss_sum / (j+1) 
        print(f"loss={eval_loss_mean:.4f}, acc={eval_acc_sum/(j+1):.4f}")
        json_dict = {f"{model_id}_vs_{other_model_id}": {
                            "loss": eval_loss_mean, 
                           "acc": eval_acc_sum / (j+1)}}
        self.save_result(f"linear_prob/{model_id}/result_{mode}.json", json_dict)
        return json_dict

    def _collect_nominal_trajectories(self, args, puffer_env, policy):
        driver = puffer_env.driver_env
        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]
        num_partners = 31
        other_trajectories = {
            "other_x": np.zeros((num_agents, num_partners, self.sim_steps), dtype=np.float32),
            "other_y": np.zeros((num_agents, num_partners, self.sim_steps), dtype=np.float32),
            "other_heading": np.zeros((num_agents, num_partners, self.sim_steps), dtype=np.float32),
            "other_id": np.zeros((num_agents, num_partners, self.sim_steps), dtype=np.int32),
            "ego_id": np.zeros((num_agents, self.sim_steps), dtype=np.int32),
            "other_speed": np.zeros((num_agents, num_partners, self.sim_steps), dtype=np.float32),
            "partner_obs": np.zeros((num_agents, num_partners, 7, self.sim_steps), dtype=np.float32),
        }
        trajectories = {
            "ego_x": np.zeros((num_agents, self.sim_steps), dtype=np.float32),
            "ego_y": np.zeros((num_agents, self.sim_steps), dtype=np.float32),
            "ego_heading": np.zeros((num_agents, self.sim_steps), dtype=np.float32),
        }

        obs, info = puffer_env.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        for time_idx in range(self.sim_steps):
            # Get global state
            partner_state = driver.get_global_partner_state()
            other_trajectories["other_x"][:, :, time_idx] = partner_state["x"]
            other_trajectories["other_y"][:, :, time_idx] = partner_state["y"]
            other_trajectories["other_speed"][:, :, time_idx] = partner_state["speed"]
            other_trajectories["other_heading"][:, :, time_idx] = partner_state["heading"]
            other_trajectories["other_id"][:, :, time_idx] = partner_state["other_id"]
            other_trajectories["ego_id"][:, time_idx] = partner_state["ego_id"]
            partner_obs = obs[:, 7:224].reshape(num_agents, 31, 7)
            other_trajectories["partner_obs"][..., time_idx] = partner_obs
            agent_state = driver.get_global_agent_state()
            trajectories["ego_x"][:, time_idx] = agent_state["x"]
            trajectories["ego_y"][:, time_idx] = agent_state["y"]
            trajectories["ego_heading"][:, time_idx] = agent_state["heading"]

            # Step policy
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

            obs, _, _, _, _ = puffer_env.step(action_np)

        return trajectories, other_trajectories

    def _collect_replay_trajectories(self, args, puffer_env, policy):

        driver = puffer_env.driver_env
        device = args["train"]["device"]
        num_partners = 31
        obs, infos = puffer_env.reset()
        ego_indices = []
        # collect the first indec in each map, set them as ego
        for i, info in enumerate(infos):
            agent_offsets = info['agent_offsets']
            ego_idx = agent_offsets[:-1]
            ego_indices += [idx + args["env"]["num_agents"] * i for idx in ego_idx]
        other_mask = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        other_mask[ego_indices] = False
        state_ego = dict(
        lstm_h=torch.zeros(len(ego_indices), policy.hidden_size, device=device),
        lstm_c=torch.zeros(len(ego_indices), policy.hidden_size, device=device),
        )
        mode = args["load_multiple_model_path"][1].split("/")
        other_action_npy = np.load(f"/data/puffer/experiments/{mode[4]}/other_action_buffer_lp/other_actions_{args['load_multiple_model_path'][1][-11:-3]}.npy")
        other_trajectories = {
            "other_x": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_y": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_heading": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_id": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.int32),
            "ego_id": np.zeros((len(ego_indices), self.sim_steps), dtype=np.int32),
            "other_speed": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "partner_obs": np.zeros((len(ego_indices), num_partners, 7, self.sim_steps), dtype=np.float32),
        }
        trajectories = {
            "ego_x": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
            "ego_y": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
            "ego_heading": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
        }
        for time_idx in range(self.sim_steps):
            # Get global state
            ob_ego = obs[ego_indices]
            partner_state = driver.get_global_partner_state()
            other_trajectories["other_x"][:, :, time_idx] = partner_state["x"][ego_indices]
            other_trajectories["other_y"][:, :, time_idx] = partner_state["y"][ego_indices]
            other_trajectories["other_speed"][:, :, time_idx] = partner_state["speed"][ego_indices]
            other_trajectories["other_heading"][:, :, time_idx] = partner_state["heading"][ego_indices]
            other_trajectories["other_id"][:, :, time_idx] = partner_state["other_id"][ego_indices]
            other_trajectories["ego_id"][:, time_idx] = partner_state["ego_id"][ego_indices]
            partner_obs = ob_ego[:, 7:224].reshape(len(ego_indices), 31, 7)
            other_trajectories["partner_obs"][..., time_idx] = partner_obs
            agent_state = driver.get_global_agent_state()
            trajectories["ego_x"][:, time_idx] = agent_state["x"][ego_indices]
            trajectories["ego_y"][:, time_idx] = agent_state["y"][ego_indices]
            trajectories["ego_heading"][:, time_idx] = agent_state["heading"][ego_indices]
            with torch.no_grad():
                total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                # ego action
                ob_ego = ob_tensor[ego_indices]
                logits_ego, value_ego = policy.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()
            
            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            total_actions[other_mask.cpu().numpy()] = other_action_npy[:, time_idx]
            total_actions[ego_indices] = action_ego
            obs, _, _, _, _  = puffer_env.step(total_actions)

            
        return trajectories, other_trajectories

            
    def _collect_reactive_trajectories(self, args, puffer_env, policy1, policy2):

        driver = puffer_env.driver_env
        device = args["train"]["device"]
        num_partners = 31
        obs, infos = puffer_env.reset()
        ego_indices = []
        # collect the first indec in each map, set them as ego
        for i, info in enumerate(infos):
            agent_offsets = info['agent_offsets']
            ego_idx = agent_offsets[:-1]
            ego_indices += [idx + args["env"]["num_agents"] * i for idx in ego_idx]
        other_mask = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        other_mask[ego_indices] = False
        state_ego = dict(
        lstm_h=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        lstm_c=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        )
        state_other = dict(
        lstm_h=torch.zeros(obs.shape[0]- len(ego_indices), policy2.hidden_size, device=device),
        lstm_c=torch.zeros(obs.shape[0] - len(ego_indices), policy2.hidden_size, device=device),
        )
        other_trajectories = {
            "other_x": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_y": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_heading": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "other_id": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.int32),
            "ego_id": np.zeros((len(ego_indices), self.sim_steps), dtype=np.int32),
            "other_speed": np.zeros((len(ego_indices), num_partners, self.sim_steps), dtype=np.float32),
            "partner_obs": np.zeros((len(ego_indices), num_partners, 7, self.sim_steps), dtype=np.float32),
        }
        trajectories = {
            "ego_x": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
            "ego_y": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
            "ego_heading": np.zeros((len(ego_indices), self.sim_steps), dtype=np.float32),
        }

        for time_idx in range(self.sim_steps):
            # Get global state
            ob_ego = obs[ego_indices]
            partner_state = driver.get_global_partner_state()
            other_trajectories["other_x"][:, :, time_idx] = partner_state["x"][ego_indices]
            other_trajectories["other_y"][:, :, time_idx] = partner_state["y"][ego_indices]
            other_trajectories["other_speed"][:, :, time_idx] = partner_state["speed"][ego_indices]
            other_trajectories["other_heading"][:, :, time_idx] = partner_state["heading"][ego_indices]
            other_trajectories["other_id"][:, :, time_idx] = partner_state["other_id"][ego_indices]
            other_trajectories["ego_id"][:, time_idx] = partner_state["ego_id"][ego_indices]
            partner_obs = ob_ego[:, 7:224].reshape(len(ego_indices), 31, 7)
            other_trajectories["partner_obs"][..., time_idx] = partner_obs
            agent_state = driver.get_global_agent_state()
            trajectories["ego_x"][:, time_idx] = agent_state["x"][ego_indices]
            trajectories["ego_y"][:, time_idx] = agent_state["y"][ego_indices]
            trajectories["ego_heading"][:, time_idx] = agent_state["heading"][ego_indices]
            with torch.no_grad():
                total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                ob_ego = ob_tensor[ego_indices]
                logits_ego, value_ego = policy1.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()

                # other action
                ob_other = ob_tensor[other_mask]
                logits_other, value_other = policy2.forward_eval(ob_other, state_other)
                action_other, logprob_other, _ = pufferlib.pytorch.sample_logits(logits_other)
                action_other = action_other.cpu().numpy()
            
            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            if isinstance(logits_other, torch.distributions.Normal):  
                action_other = np.clip(action_other, puffer_env.action_space.low, puffer_env.action_space.high)
            total_actions[ego_indices] = action_ego
            total_actions[other_mask.cpu().numpy()] = action_other
            obs, _, _, _, _ = puffer_env.step(total_actions)
            
        return trajectories, other_trajectories
    
    def make_dataset(self, args, puffer_env, policy1, policy2=None):
        model_id = args["load_multiple_model_path"][0][-11:-3]
        base_path = f"linear_prob/{model_id}"
        if args["lp_mode"]  == "generate_replay":
            other_id = args["load_multiple_model_path"][1][-11:-3]
            ego_traj, other_traj = self._collect_replay_trajectories(args, puffer_env, policy1)
            base_path = os.path.join(base_path, other_id, "replay")
        elif args["lp_mode"] == "generate_reactive":
            other_id = args["load_multiple_model_path"][1][-11:-3]
            ego_traj, other_traj = self._collect_reactive_trajectories(args, puffer_env, policy1, policy2)
            base_path = os.path.join(base_path, other_id, "reactive")
        elif args["lp_mode"] == "generate":
            ego_traj, other_traj = self._collect_nominal_trajectories(args, puffer_env, policy1)
        else:
            raise f"args.lp_mode should match in generation mode, should be in generate, generate_replay, generate_reactive, but given {args.lp_mode}"
        os.makedirs(base_path, exist_ok=True)
        future_steps = [10, 20, 30, 40]
        G = 6 # Num of grid
        lo, hi = -25, 25 # meter
        other_ids = other_traj["other_id"]          # (A,P,T)
        other_x   = other_traj["other_x"]           # (A,P,T)
        other_y   = other_traj["other_y"]           # (A,P,T)
        partner_obs = other_traj["partner_obs"]
        ego_x     = ego_traj["ego_x"]             # (A,T)
        ego_y     = ego_traj["ego_y"]             # (A,T)
        ego_h     = ego_traj["ego_heading"]       # (A,T)  (rad)
        for future_step in future_steps:
            F = future_step
            # others' future
            ids_f = other_ids[...,  F:]          # (A,P,T0)
            x_f   = other_x  [...,  F:]
            y_f   = other_y  [...,  F:]
            
            # match the ids + check existence in future
            ids_t = other_ids[..., :-F]
            eq = (ids_t[:, :, None, :] == ids_f[:, None, :, :])
            has_match = eq.any(axis=2)                       # (A,P,T0)
            idx = eq.argmax(axis=2)
            valid = (ids_t != -1) & has_match  

            # ego's current
            ego_x_t = ego_x[:, None, :-F] 
            ego_y_t = ego_y[:, None, :-F]
            partner_obs_t = partner_obs[..., :-F]
            x_f_match = np.take_along_axis(x_f, idx, axis=1) # (A,P,T0)
            y_f_match = np.take_along_axis(y_f, idx, axis=1)

            # distance between ego's current and other's future
            dx = x_f_match - ego_x_t 
            dy = y_f_match - ego_y_t

            cos_h = np.cos(ego_h[:, None, :-F])
            sin_h = np.sin(ego_h[:, None, :-F])

            rel_x = dx * cos_h + dy * sin_h
            rel_y = -dx * sin_h + dy * cos_h

            xv = rel_x[valid]
            yv = rel_y[valid]
            pobs = np.transpose(partner_obs_t, (0, 1, 3, 2)) 
            pv = pobs[valid]
            xv = np.asarray(rel_x)[np.asarray(valid)].ravel()
            yv = np.asarray(rel_y)[np.asarray(valid)].ravel()
            # ignore outside of range
            in_range = (xv >= lo) & (xv < hi) & (yv >= lo) & (yv < hi)
            x = xv[in_range]
            y = yv[in_range]    
            p_obs = pv[in_range]
            ix = np.floor((x - lo) / (hi - lo) * G).astype(np.int64)
            iy = np.floor((y - lo) / (hi - lo) * G).astype(np.int64)
            ix = np.clip(ix, 0, G-1)
            iy = np.clip(iy, 0, G-1)
            label = G * iy + ix
            save_path = os.path.join(base_path, f"future_{future_step}.npz")
            np.savez_compressed(save_path, label=label, partner_obs=p_obs)
            print(f"future step: {future_step} data saved!!! {ix.shape}")