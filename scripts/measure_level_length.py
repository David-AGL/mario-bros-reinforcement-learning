"""Mide level_length (x_pos al tomar la bandera) jugando a mano y lo guarda en levels.json.

Uso (necesita ventana: WSL2 con WSLg en Windows 11, o Windows/Linux con pantalla):
    python scripts/measure_level_length.py --level SuperMarioBros-1-1-v0
Juega con el teclado hasta tomar la bandera; el script imprime y guarda x_pos en ese momento.
Repite para cada nivel (2-1, 3-1, y 1-2 si aplica). Con --no-write solo imprime.

Teclas (por defecto de nes-py): flechas = mover, ESPACIO = salto (A), 'z'/'x' pueden usarse según el mapeo.
Nota: en niveles con tuberías o cambio de área (p. ej. 1-2) x_pos puede no crecer de forma continua;
verifícalo antes de usar progress allí (el plan sugiere reportar solo tasa de completado si no es fiable).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS_JSON = ROOT / "levels.json"


def save_level_length(level: str, x_pos: int, path: Path = LEVELS_JSON) -> bool:
    data = json.loads(path.read_text())
    for section, levels in data.items():
        if isinstance(levels, dict) and level in levels:
            levels[level]["level_length"] = int(x_pos)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            return True
    return False


class FlagWatcher:
    """Registra el x_pos en el primer paso con flag_get. Recibe el dict `info` de cada env.step."""

    def __init__(self, level: str, write: bool):
        self.level, self.write = level, write
        self.max_x, self.done_flag = 0, False

    def __call__(self, info):
        if not isinstance(info, dict):
            return
        self.max_x = max(self.max_x, int(info.get("x_pos", 0)))
        if info.get("flag_get") and not self.done_flag:
            self.done_flag = True
            x = int(info["x_pos"])
            print(f"\n>>> ¡Bandera! x_pos={x} (max_x visto={self.max_x}) en {self.level}")
            if self.write:
                ok = save_level_length(self.level, x)
                print(f">>> guardado en levels.json" if ok else ">>> nivel no encontrado en levels.json")
            print(">>> Cierra la ventana o pulsa ESC para terminar.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="SuperMarioBros-1-1-v0")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    import gym_super_mario_bros
    from nes_py.app.play_human import play_human

    env = gym_super_mario_bros.make(args.level).unwrapped  # API antigua de nes-py
    watcher = FlagWatcher(args.level, write=not args.no_write)

    # play_human NO pasa `info` a su callback (descarta el 4.º valor de env.step), así que
    # interceptamos env.step para leer info en cada paso.
    original_step = env.step

    def step_with_watch(action):
        obs, reward, done, info = original_step(action)
        watcher(info)
        return obs, reward, done, info

    env.step = step_with_watch
    try:
        play_human(env)
    finally:
        env.close()
    if not watcher.done_flag:
        print(f"No se tomó la bandera. max_x visto = {watcher.max_x} (no se guardó nada).")


if __name__ == "__main__":
    main()