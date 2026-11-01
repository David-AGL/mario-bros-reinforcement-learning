"""Entrenamiento PPO (Stable-Baselines3) para Mario Bros, con varios entornos en paralelo.

Mismo patrón que dqn/train.py: config.json por corrida, episodes.csv (max_x REAL por episodio),
--models-root / --logs-root, --resume, --dry-run.

Uso:
    python -m ppo.train --seed 1 --total-steps 50000 --run-name ppo_smoke --dry-run
    python -m ppo.train --seed 1 --total-steps 5000000 --n-envs 4 --run-name ppo_s1 \\
        --models-root /kaggle/working/out/models --logs-root /kaggle/working/out/runs
    python -m ppo.train --seed 1 --total-steps 5000000 --run-name ppo_s1 --resume <checkpoint.zip>

Receta por defecto ("v3"): recompensa x0.01 (solo para el algoritmo; las métricas siguen crudas),
rollout ~2048 transiciones, batch 256, 4 épocas, lr 2.5e-4 con decaimiento lineal a 10 %,
clip 0.2 con decaimiento lineal a 0.1. Son valores de partida, no garantías.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dqn.train import MarioDQNCallback, git_commit, versions  # noqa: E402  (callback genérico por episodio)
from env import make_env  # noqa: E402
from wrappers import ScaleReward  # noqa: E402


@dataclass
class PPOConfig:
    # Entorno
    level: str = "SuperMarioBros-1-1-v0"
    noop_max: int = 30
    seed: int = 1
    total_steps: int = 3_000_000
    n_envs: int = 8  # = núcleos de CPU disponibles (el emulador es el cuello de botella)
    reward_scale: float = 0.01  # solo lo ve el algoritmo; las métricas guardadas son crudas
    # PPO
    learning_rate: float = 2.5e-4
    lr_final_fraction: float = 0.1  # lr final = learning_rate * fraction; 1.0 = constante
    n_steps: int = 0  # por entorno; 0 = automático (rollout ~2048 transiciones en total)
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_final_fraction: float = 0.5  # clip final = clip_range * fraction; 1.0 = constante
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Infraestructura
    checkpoint_freq: int = 250_000  # en pasos totales (todos los entornos)
    device: str = "auto"
    run_name: str = ""
    models_root: str = ""  # checkpoints + config.json -> <models_root>/ppo/<run-name>/
    logs_root: str = ""  # TensorBoard + episodes.csv -> <logs_root>/ppo/<run-name>/


def make_thunk(cfg: PPOConfig):
    def _thunk():
        env = Monitor(make_env(cfg.level, noop_max=cfg.noop_max))  # recompensa cruda en info["episode"]
        if cfg.reward_scale != 1.0:
            env = ScaleReward(env, cfg.reward_scale)
        return env
    return _thunk


def schedule(start: float, final_fraction: float):
    """Constante si final_fraction >= 1; si no, decaimiento lineal start -> start*final_fraction."""
    if final_fraction >= 1.0:
        return start
    return get_linear_fn(start, start * final_fraction, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrenar PPO en Mario Bros")
    d = PPOConfig()
    for name, val in asdict(d).items():
        flag = "--" + name.replace("_", "-")
        p.add_argument(flag, type=type(val), default=val)
    p.add_argument("--resume", type=str, default=None, help="checkpoint .zip a reanudar")
    p.add_argument("--dry-run", action="store_true", help="imprime config y sale")
    args = p.parse_args(argv)

    cfg = PPOConfig(**{k: getattr(args, k) for k in asdict(d)})
    cfg.run_name = cfg.run_name or f"ppo_s{cfg.seed}"
    if cfg.n_steps <= 0:
        cfg.n_steps = max(1, 2048 // cfg.n_envs)
    rollout = cfg.n_steps * cfg.n_envs
    models_base = Path(cfg.models_root) if cfg.models_root else ROOT / "models"
    logs_base = Path(cfg.logs_root) if cfg.logs_root else ROOT / "runs"
    model_dir = models_base / "ppo" / cfg.run_name
    log_dir = logs_base / "ppo" / cfg.run_name

    print(f"Config: {json.dumps(asdict(cfg), indent=2)}")
    print(f"Rollout = {cfg.n_steps} x {cfg.n_envs} = {rollout} transiciones; "
          f"minibatches por época = {rollout / cfg.batch_size:.1f}")
    print(f"CPU: {os.cpu_count()} núcleos | RAM libre: {psutil.virtual_memory().available / 1e9:.1f} GB")
    if cfg.n_envs > (os.cpu_count() or 1):
        print("AVISO: n_envs supera los núcleos de CPU; no acelerará el entrenamiento.")
    if rollout % cfg.batch_size:
        print("AVISO: batch_size no divide el rollout; SB3 truncará el último minibatch.")
    if args.dry_run:
        return

    if not args.resume and model_dir.exists() and any(model_dir.iterdir()):
        sys.exit(f"{model_dir} ya existe y no está vacío. Usa otro --run-name o --resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "config": asdict(cfg), "git_commit": git_commit(), "versions": versions(),
        "cpu_count": os.cpu_count(), "resumed_from": args.resume}, indent=2))

    venv = SubprocVecEnv([make_thunk(cfg) for _ in range(cfg.n_envs)])
    venv.seed(cfg.seed)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=venv, device=cfg.device)
            remaining = cfg.total_steps - model.num_timesteps
            reset_ts = False
        else:
            model = PPO(
                "CnnPolicy", venv,
                learning_rate=schedule(cfg.learning_rate, cfg.lr_final_fraction),
                n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
                gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                clip_range=schedule(cfg.clip_range, cfg.clip_final_fraction),
                ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
                tensorboard_log=str(logs_base / "ppo"), seed=cfg.seed, device=cfg.device, verbose=1,
            )
            remaining = cfg.total_steps
            reset_ts = True
        if remaining <= 0:
            sys.exit("El modelo ya alcanzó total_steps.")

        callbacks = CallbackList([
            CheckpointCallback(save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
                               save_path=str(model_dir), name_prefix=f"ppo_s{cfg.seed}"),
            MarioDQNCallback(log_dir / "episodes.csv", verbose=1),
        ])
        model.learn(total_timesteps=remaining, callback=callbacks, tb_log_name=cfg.run_name,
                    reset_num_timesteps=reset_ts, log_interval=10)
        model.save(model_dir / f"ppo_s{cfg.seed}_final")
        print(f"Modelo final: {model_dir / f'ppo_s{cfg.seed}_final.zip'}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
