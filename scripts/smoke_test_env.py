"""Smoke test del entorno (puerta de entrada del día 1). Falla con AssertionError si algo no cuadra."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from env import make_env, N_STACK, FRAME_SIZE

LEVELS = ["SuperMarioBros-1-1-v0", "SuperMarioBros-2-1-v0", "SuperMarioBros-3-1-v0", "SuperMarioBros-1-2-v0"]

def check_obs(obs, env):
    assert obs.shape == (N_STACK, FRAME_SIZE, FRAME_SIZE), obs.shape
    assert obs.dtype == np.uint8, obs.dtype
    assert 0 <= obs.min() and obs.max() <= 255
    assert env.observation_space.contains(obs)

def run(env_id, noop_max, n_steps=2000, seed=0):
    env = make_env(env_id, noop_max=noop_max)
    obs, info = env.reset(seed=seed)
    check_obs(obs, env)
    assert 0 <= info["noop_used"] <= noop_max
    episodes, max_x, t0 = 0, 0, time.time()
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        check_obs(obs, env)
        max_x = max(max_x, info.get("x_pos", 0))
        if term or trunc:
            episodes += 1
            obs, info = env.reset()
    fps = n_steps / (time.time() - t0)
    env.close()
    return episodes, max_x, fps

if __name__ == "__main__":
    for lvl in LEVELS:
        eps, mx, fps = run(lvl, noop_max=30)
        print(f"{lvl}: OK | episodios={eps} max_x={mx} FPS(agent steps)={fps:.0f}")
    e = make_env(noop_max=30)
    a = [e.reset(seed=200)[1]["noop_used"] for _ in range(3)]
    assert len(set(a)) == 1, a
    print("No-ops reproducibles con la misma semilla:", a)
    print("SMOKE TEST OK")
