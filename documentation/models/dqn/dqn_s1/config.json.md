# `models/dqn/dqn_s1/config.json`

Metadatos de la corrida `dqn_s1`.

- `config`: parámetros efectivos del entorno y DQN, presupuesto de 50000 pasos, semilla 1, nivel `1-1`, `noop_max=30`, replay buffer optimizado y rutas de Colab.
- `git_commit`: commit desde el que se entrenó el modelo.
- `versions`: versiones de Python, Torch, SB3, Gymnasium, NumPy y disponibilidad de CUDA.
- `buffer_estimate_gb`: estimación del buffer usando la configuración de memoria.
- `resumed_from`: checkpoint de origen; `null` indica que la corrida comenzó desde cero.

Este archivo permite reproducir o auditar el contexto de un checkpoint sin inferirlo desde el nombre del archivo.

