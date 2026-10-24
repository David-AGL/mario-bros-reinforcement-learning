"""Entrenamiento DQN (Stable-Baselines3) para Mario Bros — esqueleto del día 1.

Uso típico:
    # 1) Ver configuración + estimación de memoria sin entrenar
    python -m dqn.train --seed 1 --total-steps 50000 --dry-run
    # 2) Smoke test (día 2): 50k pasos
    python -m dqn.train --seed 1 --total-steps 50000 --run-name dqn_smoke
    # 3) Reanudar tras una caída (p. ej. Colab)
    python -m dqn.train --seed 1 --total-steps 1000000 --resume models/dqn/dqn_s1/dqn_s1_300000_steps.zip
    # 4) Colab: checkpoints en Drive, logs en disco local
    python -m dqn.train --seed 1 --total-steps 1000000 \\
        --models-root /content/drive/MyDrive/mario/models --logs-root /content/runs

Hiperparámetros por defecto = tabla 4 del plan (valores de partida, no garantías).
Salidas:
    models/dqn/<run-name>/      checkpoints (.zip), final, config.json
    runs/dqn/<run-name>/        TensorBoard + episodes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env import ENV_VERSION, FRAME_SIZE, N_STACK, make_env  # noqa: E402


# --------------------------------------------------------------------------- config
@dataclass
class DQNConfig:
    # Entorno
    level: str = "SuperMarioBros-1-1-v0"
    noop_max: int = 30  # DECISIÓN A ACORDAR: no-ops aleatorios también en entrenamiento
    seed: int = 1
    total_steps: int = 1_000_000  # presupuesto: fijar tras medir FPS (igual para DQN y PPO)
    # DQN (tabla 4 del plan)
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 20_000
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.2  # fracción de total_steps
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    optimize_memory_usage: bool = False  # True: ~mitad de RAM; ver estimate_buffer_gb
    # Infraestructura
    checkpoint_freq: int = 100_000
    save_replay_buffer: bool = False  # el buffer pesa GBs; sin él, al reanudar se re-llena
    device: str = "auto"
    run_name: str = ""
    # Rutas (vacío = carpetas del repo). En Colab: models_root en Drive, logs_root en disco local
    models_root: str = ""  # checkpoints + config.json -> <models_root>/dqn/<run-name>/
    logs_root: str = ""  # TensorBoard + episodes.csv -> <logs_root>/dqn/<run-name>/


def estimate_buffer_gb(buffer_size: int, optimize_memory_usage: bool) -> float:
    """Estimación de RAM del replay buffer (solo observaciones; el resto es despreciable).

    obs = 4*84*84 uint8 = 28 224 B. SB3 guarda obs y next_obs (x2) salvo con
    optimize_memory_usage=True, que solo guarda obs (x1).
    """
    obs_bytes = N_STACK * FRAME_SIZE * FRAME_SIZE
    copies = 1 if optimize_memory_usage else 2
    return buffer_size * obs_bytes * copies / 1e9


# --------------------------------------------------------------------------- utilidades
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    import gymnasium, stable_baselines3
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "env_version": ENV_VERSION,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import gym_super_mario_bros, nes_py
        out["gym_super_mario_bros"] = getattr(gym_super_mario_bros, "__version__", "?")
        out["nes_py"] = getattr(nes_py, "__version__", "?")
    except Exception:
        pass
    return out


class MarioDQNCallback(BaseCallback):
    """Registra por episodio max_x, flag_get y recompensa; y la RAM del proceso."""

    def __init__(self, csv_path: Path, rss_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rss_every = rss_every
        self._max_x: list[int] = []
        self._episode = 0
        self._proc = psutil.Process()
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._max_x = [0] * n_envs
        new = not self.csv_path.exists()
        self._f = open(self.csv_path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["episode", "timesteps", "max_x", "flag_get", "reward", "length", "elapsed_seconds"])

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        for i, (info, done) in enumerate(zip(infos, dones)):
            self._max_x[i] = max(self._max_x[i], int(info.get("x_pos", 0)))
            if done:
                self._episode += 1
                ep = info.get("episode", {})
                flag = bool(info.get("flag_get", False))
                self.logger.record("mario/max_x", self._max_x[i])
                self.logger.record("mario/flag_get", float(flag))
                self._w.writerow([self._episode, self.num_timesteps, self._max_x[i], int(flag),
                                  ep.get("r", ""), ep.get("l", ""), round(time.time() - self._t0, 1)])
                self._f.flush()
                self._max_x[i] = 0
        if self.num_timesteps % self.rss_every == 0:
            rss = self._proc.memory_info().rss / 1e9
            self.logger.record("system/rss_gb", rss)
            if self.verbose:
                print(f"[mem] steps={self.num_timesteps} RSS={rss:.2f} GB")
        return True

    def _on_training_end(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- main
def build_env(cfg: DQNConfig, log_dir: Path):
    def _thunk():
        env = make_env(cfg.level, noop_max=cfg.noop_max)
        return Monitor(env, filename=None)  # info["episode"] = {r, l, t}
    venv = DummyVecEnv([_thunk])
    venv.seed(cfg.seed)
    return venv


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrenar DQN en Mario Bros")
    d = DQNConfig()
    for name, val in asdict(d).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(val, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=val)
        else:
            p.add_argument(flag, type=type(val), default=val)
    p.add_argument("--resume", type=str, default=None, help="checkpoint .zip a reanudar")
    p.add_argument("--replay-buffer", type=str, default=None, help="replay buffer .pkl (con --resume)")
    p.add_argument("--dry-run", action="store_true", help="imprime config/memoria y sale")
    args = p.parse_args(argv)

    cfg = DQNConfig(**{k: getattr(args, k) for k in asdict(d)})
    cfg.run_name = cfg.run_name or f"dqn_s{cfg.seed}"
    models_base = Path(cfg.models_root) if cfg.models_root else ROOT / "models"
    logs_base = Path(cfg.logs_root) if cfg.logs_root else ROOT / "runs"
    model_dir = models_base / "dqn" / cfg.run_name
    log_dir = logs_base / "dqn" / cfg.run_name

    gb = estimate_buffer_gb(cfg.buffer_size, cfg.optimize_memory_usage)
    avail = psutil.virtual_memory().available / 1e9
    print(f"Config: {json.dumps(asdict(cfg), indent=2)}")
    print(f"Replay buffer estimado: {gb:.2f} GB | RAM disponible ahora: {avail:.1f} GB")
    if gb > 0.6 * avail:
        print("AVISO: el buffer ocuparía >60% de la RAM libre. Considera --optimize-memory-usage o bajar --buffer-size.")
    if args.dry_run:
        return

    if not args.resume and model_dir.exists() and any(model_dir.iterdir()):
        sys.exit(f"{model_dir} ya existe y no está vacío. Usa otro --run-name o --resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = {"config": asdict(cfg), "git_commit": git_commit(), "versions": versions(),
            "buffer_estimate_gb": round(gb, 2), "resumed_from": args.resume}
    (model_dir / "config.json").write_text(json.dumps(meta, indent=2))

    venv = build_env(cfg, log_dir)

    rb_kwargs = {}
    if cfg.optimize_memory_usage:
        # SB3: optimize_memory_usage es incompatible con handle_timeout_termination=True.
        # Aquí no hay truncamiento por tiempo (max_episode_steps=None), así que es seguro.
        rb_kwargs = dict(handle_timeout_termination=False)

    if args.resume:
        model = DQN.load(args.resume, env=venv, device=cfg.device)
        if args.replay_buffer:
            model.load_replay_buffer(args.replay_buffer)
        else:
            # Sin buffer guardado: volver a llenar antes de entrenar (si no, muestrea un buffer vacío)
            model.learning_starts = model.num_timesteps + cfg.learning_starts
            print(f"Reanudando sin replay buffer: learning_starts -> {model.learning_starts}")
        remaining = cfg.total_steps - model.num_timesteps
        reset_ts = False
    else:
        model = DQN(
            "CnnPolicy", venv,
            learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts, batch_size=cfg.batch_size,
            gamma=cfg.gamma, train_freq=cfg.train_freq, gradient_steps=cfg.gradient_steps,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_initial_eps=cfg.exploration_initial_eps,
            exploration_final_eps=cfg.exploration_final_eps,
            max_grad_norm=cfg.max_grad_norm,
            optimize_memory_usage=cfg.optimize_memory_usage, replay_buffer_kwargs=rb_kwargs,
            tensorboard_log=str(logs_base / "dqn"), seed=cfg.seed, device=cfg.device, verbose=1,
        )
        remaining = cfg.total_steps
        reset_ts = True

    if remaining <= 0:
        sys.exit("El modelo ya alcanzó total_steps.")

    callbacks = CallbackList([
        CheckpointCallback(save_freq=cfg.checkpoint_freq, save_path=str(model_dir),
                           name_prefix=f"dqn_s{cfg.seed}", save_replay_buffer=cfg.save_replay_buffer),
        MarioDQNCallback(log_dir / "episodes.csv", verbose=1),
    ])
    model.learn(total_timesteps=remaining, callback=callbacks, tb_log_name=cfg.run_name,
                reset_num_timesteps=reset_ts, log_interval=10)
    model.save(model_dir / f"dqn_s{cfg.seed}_final")
    print(f"Modelo final: {model_dir / f'dqn_s{cfg.seed}_final.zip'}")


if __name__ == "__main__":
    main()
