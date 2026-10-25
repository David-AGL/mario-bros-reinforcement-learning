# Mario Bros con aprendizaje por refuerzo (DQN + baseline PPO)

## Instalación
1. Instalar PyTorch según tu CUDA (https://pytorch.org).
2. `pip install -r requirements.txt`
3. `python scripts/smoke_test_env.py` (debe terminar en `SMOKE TEST OK`).

## Entorno (env-v1)
SIMPLE_MOVEMENT · frame skip 4 · gris · 84×84 · stack 4 · observación `(4, 84, 84)` uint8.
Ver `env.py`.

## Flujo híbrido
Entrenar en Colab con `notebooks/colab_train.ipynb` (checkpoints en Drive); evaluar y visualizar en local.
`dqn/train.py` acepta `--models-root` y `--logs-root` para separar dónde se guardan modelos y logs.

## Evaluación
```
python evaluate.py --algo random --seeds 100-104
python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --seeds 100-104 --out results/val
python evaluate.py --algo dqn --model ... --mode deterministic --seeds 100-104 --record
```
Protocolo: 1 episodio por semilla, no-ops 0–30 al reiniciar, DQN epsilon-greedy fijo (0.05) aplicado en
`evaluate.py`, PPO muestreando la política; `--mode deterministic` como resultado secundario.
Salida: `<out>/<tag>.csv` y `.json`. `progress` requiere `level_length` en `levels.json`.