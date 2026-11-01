"""Wrappers de entrenamiento (NO modifican env.py ni env-v2).

ScaleReward: multiplica la recompensa que ve el algoritmo. Se coloca POR FUERA de Monitor,
de modo que `info["episode"]["r"]` (y episodes.csv) siguen registrando la recompensa cruda,
y la evaluación (evaluate.py) no se ve afectada porque usa el entorno sin escalar.
"""
import gymnasium as gym


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float = 0.01):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale
