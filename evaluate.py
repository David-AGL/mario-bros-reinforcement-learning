"""Evaluación reproducible de agentes de Mario Bros (DQN, PPO o aleatorio).

Protocolo (plan, sección 1) — mismo para todos los modelos:
  * 1 episodio por semilla de evaluación (--seeds, p. ej. 100-104 validación, 200-209 final).
  * No-ops aleatorios 0..noop_max al reiniciar (sembrados con la semilla de evaluación).
  * DQN estocástico: epsilon-greedy con epsilon FIJO (--epsilon, 0.05) aplicado AQUÍ;
    el exploration_rate guardado en el modelo se ignora (siempre se llama predict(deterministic=True)).
  * PPO estocástico: muestreo de la política. Ambos con --mode deterministic: acción greedy/modo.
    Ojo: epsilon-greedy y muestreo NO son aleatoriedades equivalentes; reportar también el modo determinista.
  * Progreso = max_x / level_length (levels.json). completed = info["flag_get"] en algún paso.

Ejemplos:
    python evaluate.py --algo random --level SuperMarioBros-1-1-v0 --seeds 100-104
    python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --seeds 100-104 --out results/val
    python evaluate.py --algo dqn --model ... --mode deterministic --seeds 100-104 --record

Salida: <out>/<tag>.csv y <tag>.json (filas por episodio + resumen). `run_episode` es reutilizable (interfaz).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from env import ENV_VERSION, make_env  # noqa: E402

CSV_FIELDS = [
    "algo", "seed", "level", "episode", "completed", "max_x", "steps", "reward",
    "elapsed_seconds", "prediction_mode",
    "eval_seed", "checkpoint_steps", "noop_max", "noop_used", "level_length", "progress",
    "death_x", "end_x", "truncated", "git_commit",
]
VIDEO_FPS = 15  # 60 fps del NES / frame skip 4


# --------------------------------------------------------------------------- utilidades
def parse_seeds(text: str) -> list[int]:
    """'200-209' -> [200..209]; '1,2,5' -> [1,2,5]."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_level_length(level: str, path: Path = ROOT / "levels.json"):
    """x_pos al tomar la bandera (levels.json). None si no está medido."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    for section, levels in data.items():
        if isinstance(levels, dict) and level in levels:
            return levels[level].get("level_length")
    return None


# --------------------------------------------------------------------------- políticas
class Policy:
    """Envuelve DQN / PPO / aleatorio con una interfaz común: reset(seed) y act(obs)."""

    def __init__(self, algo: str, model=None, mode: str = "stochastic",
                 epsilon: float = 0.05, n_actions: int = 7):
        assert algo in ("dqn", "ppo", "random") and mode in ("stochastic", "deterministic")
        self.algo, self.model, self.mode = algo, model, mode
        self.epsilon, self.n_actions = float(epsilon), n_actions
        self.rng = np.random.default_rng(0)

    @property
    def prediction_mode(self) -> str:
        if self.algo == "random":
            return "uniform_random"
        if self.algo == "dqn":
            return f"epsilon_greedy_{self.epsilon:g}" if self.mode == "stochastic" else "greedy"
        return "sample" if self.mode == "stochastic" else "deterministic"

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([seed, 1])  # independiente del RNG del entorno (no-ops)
        if self.algo == "ppo" and self.mode == "stochastic":
            import torch
            torch.manual_seed(seed)  # muestreo de la política reproducible

    def act(self, obs) -> int:
        if self.algo == "random":
            return int(self.rng.integers(self.n_actions))
        if self.algo == "dqn":
            if self.mode == "stochastic" and self.rng.random() < self.epsilon:
                return int(self.rng.integers(self.n_actions))
            action, _ = self.model.predict(obs, deterministic=True)  # ignora exploration_rate
            return int(action)
        action, _ = self.model.predict(obs, deterministic=(self.mode == "deterministic"))
        return int(action)


def load_policy(algo: str, model_path: str | None, mode: str, epsilon: float, device: str = "cpu"):
    """Devuelve (policy, checkpoint_steps)."""
    if algo == "random":
        return Policy("random", mode=mode, epsilon=epsilon), 0
    if not model_path:
        sys.exit("--model es obligatorio para dqn/ppo")
    if algo == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path, device=device)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)
    return Policy(algo, model, mode, epsilon, n_actions=model.action_space.n), int(model.num_timesteps)


# --------------------------------------------------------------------------- episodio
def run_episode(env, policy: Policy, eval_seed: int, max_steps: int = 5000,
                record_path: Path | None = None) -> dict:
    """Un episodio completo. Reutilizable desde la interfaz (no duplicar lógica)."""
    obs, info = env.reset(seed=eval_seed)
    policy.reset(eval_seed)
    noop_used = int(info.get("noop_used", 0))

    writer = None
    total_reward, max_x, end_x, steps = 0.0, 0, 0, 0
    completed, truncated = False, False
    t0 = time.time()

    def write_frame():
        nonlocal writer
        frame = env.render()
        if frame is None:
            return
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        if writer is None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if record_path is not None:
        write_frame()
    while True:
        action = policy.act(obs)
        obs, reward, terminated, trunc, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        end_x = int(info.get("x_pos", 0))
        max_x = max(max_x, end_x)
        completed = completed or bool(info.get("flag_get", False))
        if record_path is not None:
            write_frame()
        if terminated or trunc:
            break
        if steps >= max_steps:
            truncated = True
            break
    if writer is not None:
        writer.release()

    return {
        "completed": int(completed), "max_x": max_x, "steps": steps, "reward": round(total_reward, 2),
        "elapsed_seconds": round(time.time() - t0, 2), "eval_seed": eval_seed, "noop_used": noop_used,
        "death_x": "" if completed else end_x, "end_x": end_x, "truncated": int(truncated),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"episodes": n, "completion_rate": sum(r["completed"] for r in rows) / n,
           "max_x_mean": float(np.mean([r["max_x"] for r in rows])),
           "max_x_std": float(np.std([r["max_x"] for r in rows]))}
    prog = [r["progress"] for r in rows if r["progress"] != ""]
    out["progress_mean"] = float(np.mean(prog)) if len(prog) == n else None
    out["progress_std"] = float(np.std(prog)) if len(prog) == n else None
    return out


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluar un agente de Mario Bros")
    p.add_argument("--algo", choices=["dqn", "ppo", "random"], required=True)
    p.add_argument("--model", type=str, default=None, help="ruta al .zip (dqn/ppo)")
    p.add_argument("--level", type=str, default="SuperMarioBros-1-1-v0")
    p.add_argument("--seeds", type=str, default="100-104", help="semillas de evaluación: '100-104' o '1,2,3'")
    p.add_argument("--mode", choices=["stochastic", "deterministic"], default="stochastic")
    p.add_argument("--epsilon", type=float, default=0.05, help="epsilon fijo para DQN estocástico")
    p.add_argument("--noop-max", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=5000, help="tope de pasos de agente por episodio")
    p.add_argument("--train-seed", type=str, default=None, help="semilla de entrenamiento (por defecto, del nombre del modelo)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="results/eval", help="carpeta de salida")
    p.add_argument("--tag", type=str, default=None, help="prefijo de los archivos de salida")
    p.add_argument("--record", action="store_true", help="grabar un mp4 por episodio en videos/")
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    policy, ckpt_steps = load_policy(args.algo, args.model, args.mode, args.epsilon, args.device)
    train_seed = args.train_seed
    if train_seed is None:
        m = re.search(r"_s(\d+)", Path(args.model).name) if args.model else None
        train_seed = m.group(1) if m else ""
    level_length = load_level_length(args.level)
    if level_length is None:
        print(f"AVISO: level_length de {args.level} no está en levels.json; progress quedará vacío.")

    env = make_env(args.level, noop_max=args.noop_max, render_mode="rgb_array" if args.record else None)
    commit = git_commit()
    level_short = args.level.replace("SuperMarioBros-", "").replace("-v0", "")
    tag = args.tag or f"{args.algo}_{level_short}_{policy.prediction_mode}_seeds{seeds[0]}-{seeds[-1]}"

    rows = []
    for i, seed in enumerate(seeds):
        rec = ROOT / "videos" / f"{tag}_seed{seed}.mp4" if args.record else None
        r = run_episode(env, policy, seed, max_steps=args.max_steps, record_path=rec)
        r.update({
            "algo": args.algo, "seed": train_seed, "level": args.level, "episode": i,
            "prediction_mode": policy.prediction_mode, "checkpoint_steps": ckpt_steps,
            "noop_max": args.noop_max, "level_length": level_length if level_length else "",
            "progress": round(r["max_x"] / level_length, 4) if level_length else "",
            "git_commit": commit,
        })
        rows.append(r)
        print(f"[{i+1}/{len(seeds)}] seed={seed} noop={r['noop_used']} max_x={r['max_x']} "
              f"completed={r['completed']} steps={r['steps']} reward={r['reward']}")
    env.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    meta = {"args": vars(args), "env_version": ENV_VERSION, "git_commit": commit,
            "prediction_mode": policy.prediction_mode, "checkpoint_steps": ckpt_steps}
    (out_dir / f"{tag}.json").write_text(json.dumps({"meta": meta, "summary": summary, "episodes": rows}, indent=2))
    print("Resumen:", json.dumps(summary))
    print(f"Guardado: {out_dir / (tag + '.csv')}")


if __name__ == "__main__":
    main()
