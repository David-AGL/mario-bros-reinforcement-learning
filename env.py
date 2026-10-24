"""Entorno de Mario Bros para RL (env-v1).

Preprocesamiento CONGELADO (cambiarlo invalida la comparabilidad entre modelos):
  - Acciones: SIMPLE_MOVEMENT (7 acciones discretas; la 0 es NOOP)
  - Frame skip: 4 (la acción se repite 4 frames y se suman las recompensas)
  - Escala de grises + resize 84x84 (INTER_AREA)
  - Stack de 4 frames -> observación (4, 84, 84) uint8 en [0, 255]
  - Reward: la del entorno (sin clipping ni escalado); se documenta en el README
  - noop_max: al reiniciar se aplican k NOOPs aleatorios, k ~ U{0..noop_max}
    (k se registra en info["noop_used"]). noop_max=0 desactiva el mecanismo.

El entorno es un gymnasium.Env que envuelve directamente el entorno nes-py
(API antigua de gym), de modo que SB3 2.x lo usa sin shims.
"""
from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

ENV_VERSION = "env-v1"
FRAME_SKIP = 4
FRAME_SIZE = 84
N_STACK = 4
NOOP_ACTION = 0  # índice de ['NOOP'] en SIMPLE_MOVEMENT


def level_id(world: int, stage: int, version: str = "v0") -> str:
    """ID del entorno, p. ej. SuperMarioBros-1-1-v0. Confirmar con `gym_super_mario_bros.make`."""
    return f"SuperMarioBros-{world}-{stage}-{version}"


class MarioEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str = "SuperMarioBros-1-1-v0",
        noop_max: int = 0,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.env_id = env_id
        self.noop_max = int(noop_max)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # .unwrapped salta los wrappers de gym 0.26 (TimeLimit, PassiveEnvChecker...) que
        # asumen la API nueva; nes-py usa la API antigua (reset->obs, step->4 valores).
        base = gym_super_mario_bros.make(env_id).unwrapped
        self._nes = JoypadSpace(base, SIMPLE_MOVEMENT)
        assert SIMPLE_MOVEMENT[NOOP_ACTION] == ["NOOP"]

        self.action_space = spaces.Discrete(len(SIMPLE_MOVEMENT))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(N_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8
        )
        self._frames: deque[np.ndarray] = deque(maxlen=N_STACK)
        self._steps = 0
        self._last_raw = None

    # ---------- utilidades ----------
    @staticmethod
    def _process(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)

    def _obs(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    # ---------- API gymnasium ----------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # siembra self.np_random (se usa para los no-ops)
        raw = self._nes.reset()
        self._last_raw = raw
        self._steps = 0

        noop_used = 0
        info = {}
        if self.noop_max > 0:
            noop_used = int(self.np_random.integers(0, self.noop_max + 1))
            for _ in range(noop_used):
                raw, _, done, info = self._nes.step(NOOP_ACTION)
                self._last_raw = raw
                if done:  # muy improbable; reiniciar sin no-ops
                    raw = self._nes.reset()
                    self._last_raw = raw
                    noop_used = 0
                    info = {}
                    break

        frame = self._process(raw)
        for _ in range(N_STACK):
            self._frames.append(frame)

        info = dict(info)
        info["noop_used"] = noop_used
        info["env_id"] = self.env_id
        return self._obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        info = {}
        raw = self._last_raw
        for _ in range(FRAME_SKIP):
            raw, r, done, info = self._nes.step(int(action))
            total_reward += float(r)
            if done:
                terminated = True
                break
        self._last_raw = raw
        self._frames.append(self._process(raw))
        self._steps += 1

        truncated = bool(
            not terminated
            and self.max_episode_steps is not None
            and self._steps >= self.max_episode_steps
        )
        return self._obs(), total_reward, terminated, truncated, dict(info)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._last_raw
        return None

    def close(self):
        self._nes.close()


def make_env(env_id: str = "SuperMarioBros-1-1-v0", noop_max: int = 0, **kwargs) -> MarioEnv:
    return MarioEnv(env_id=env_id, noop_max=noop_max, **kwargs)
