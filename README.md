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
