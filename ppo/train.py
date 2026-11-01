"""Entrenamiento PPO (Stable-Baselines3) para Mario Bros con múltiples entornos paralelos.

Uso típico:
    python -m ppo.train --seed 1 --total-steps 2000000 --n-envs 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env import ENV_VERSION, make_env


@dataclass
class PPOConfig:
    level: str = "SuperMarioBros-1-1-v0"
    noop_max: int = 30
    seed: int = 1
    total_steps: int = 2_000_000
    n_envs: int = 8
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    checkpoint_freq: int = 100_000
    device: str = "auto"
    run_name: str = "ppo_s1"


class MarioPPOCallback(BaseCallback):
    def __init__(self, log_csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_csv_path = Path(log_csv_path)
        self.file = None
        self.writer = None
        self.episode_count = 0

    def _on_training_start(self) -> None:
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_csv_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["episode", "timesteps", "reward", "length", "max_x", "flag_get"]
        )
        self.writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                max_x = int(info.get("x_pos", 0))
                flag_get = int(bool(info.get("flag_get", False)))
                self.writer.writerow({
                    "episode": self.episode_count,
                    "timesteps": self.num_timesteps,
                    "reward": round(info["episode"]["r"], 2),
                    "length": info["episode"]["l"],
                    "max_x": max_x,
                    "flag_get": flag_get
                })
                self.file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file:
            self.file.close()


def make_single_env(env_id: str, rank: int, seed: int = 0, noop_max: int = 30):
    def _thunk():
        env = make_env(env_id, noop_max=noop_max)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser(description="Entrenar PPO en Super Mario Bros")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--level", type=str, default="SuperMarioBros-1-1-v0")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--models-root", type=str, default="./models/ppo")
    parser.add_argument("--logs-root", type=str, default="./runs/ppo")
    parser.add_argument("--run-name", type=str, default="ppo_s1")
    args = parser.parse_args()

    env_fns = [make_single_env(args.level, i, seed=args.seed) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    models_dir = Path(args.models_root) / args.run_name
    logs_dir = Path(args.logs_root) / args.run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000, 100_000 // args.n_envs),
        save_path=str(models_dir),
        name_prefix="ppo_mario"
    )
    mario_cb = MarioPPOCallback(logs_dir / "episodes.csv", verbose=1)

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(logs_dir),
        seed=args.seed,
        verbose=1
    )

    print(f"Iniciando entrenamiento PPO con {args.n_envs} entornos paralelos...")
    model.learn(total_timesteps=args.total_steps, callback=CallbackList([checkpoint_cb, mario_cb]))
    model.save(str(models_dir / "ppo_mario_final"))
    print("Entrenamiento PPO completado.")
    vec_env.close()


if __name__ == "__main__":
    main()
