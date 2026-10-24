# `dqn/train.py`

Punto de entrada del entrenamiento DQN con Stable-Baselines3. Se puede ejecutar como `python -m dqn.train`.

## Importaciones y configuración del módulo

El módulo importa herramientas de línea de comandos (`argparse`), serialización (`json`), información del sistema (`platform`, `psutil`), ejecución de Git (`subprocess`), tiempos y archivos (`time`, `Path`), y componentes de Stable-Baselines3. `numpy` y `torch` se usan para registrar versiones y seleccionar CUDA.

`ROOT = Path(__file__).resolve().parents[1]` apunta a la raíz del repositorio aunque el comando se ejecute desde otra carpeta. El `sys.path.insert` permite importar `env.py` desde el módulo `dqn`.

## `DQNConfig`

Dataclass con la configuración completa:

- nivel, semilla, no-ops y presupuesto total;
- hiperparámetros de DQN;
- frecuencia de checkpoints, replay buffer y dispositivo;
- nombre de corrida y raíces separadas para modelos y logs.

Los valores por defecto corresponden al experimento inicial. `run_name` vacío se convierte en `dqn_s<seed>`.

Cada campo se convierte automáticamente en una opción CLI: por ejemplo, `total_steps` se expone como `--total-steps`, `learning_rate` como `--learning-rate` y los booleanos admiten `--optimize-memory-usage`/`--no-optimize-memory-usage`. Esto evita duplicar la lista de hiperparámetros en el parser.

## Utilidades

- `estimate_buffer_gb`: estima solo la memoria de observaciones del replay buffer y tiene en cuenta `optimize_memory_usage`.
- `git_commit`: guarda el commit corto en los metadatos, o `unknown` si no puede consultarlo.
- `versions`: recopila versiones de Python, PyTorch, SB3, Gymnasium, NumPy, el entorno y disponibilidad de CUDA.

`git_commit` captura el estado del código, mientras que `versions` captura el estado del entorno Python. Juntos permiten saber si un checkpoint fue producido con una combinación distinta de dependencias.

## `MarioDQNCallback`

Callback personalizado que:

- acumula el máximo `x_pos` de cada episodio;
- registra `flag_get`, recompensa, longitud y tiempo transcurrido;
- escribe episodios en `episodes.csv`;
- publica métricas `mario/max_x`, `mario/flag_get` y `system/rss_gb` en TensorBoard.

El RSS se mide periódicamente con `psutil`.

### Flujo interno del callback

- `_on_training_start` obtiene el número de entornos vectorizados, crea un acumulador `max_x` por entorno y abre `episodes.csv` en modo append. Escribe la cabecera solo si el archivo es nuevo.
- `_on_step` lee `infos` y `dones` que SB3 deja en `self.locals`. Actualiza el máximo `x_pos` de cada entorno. Cuando termina un episodio, toma las métricas insertadas por `Monitor` (`r`, `l`, `t`), registra las métricas de TensorBoard, escribe una fila y reinicia el acumulador de ese entorno.
- La medición de RSS ocurre cuando `num_timesteps` es múltiplo de `rss_every`; se guarda en gigabytes y opcionalmente se imprime.
- `_on_training_end` cierra el archivo CSV para no perder buffers de escritura.

## Construcción y ejecución

- `build_env` crea `MarioEnv`, lo envuelve en `Monitor` y `DummyVecEnv`, y aplica la semilla.
- `main` crea argumentos automáticamente desde `DQNConfig`, añade `--resume`, `--replay-buffer` y `--dry-run`, estima memoria y muestra la configuración.
- En una corrida nueva, crea `models/dqn/<run-name>` y `runs/dqn/<run-name>`, guarda `config.json`, construye `DQN("CnnPolicy", ...)` y comienza `learn`.
- `CheckpointCallback` guarda checkpoints periódicos; al final se guarda `<run-name>_final.zip`.
- Con `--resume` carga un checkpoint y continúa solo los pasos restantes. Si no se proporciona replay buffer, retrasa `learning_starts` para rellenarlo nuevamente.

El script aborta si la carpeta de la corrida ya existe y no se solicitó reanudar, y también si el modelo ya alcanzó `total_steps`.

## Flujo detallado de `main`

1. Crea el parser y obtiene los valores predeterminados de `DQNConfig` con `asdict`.
2. Calcula las rutas: por defecto `models/dqn/<run_name>` y `runs/dqn/<run_name>`, o las raíces especificadas por Colab.
3. Imprime el JSON de configuración y la estimación de memoria. Si el buffer supera el 60% de la RAM libre, muestra una advertencia. `--dry-run` termina aquí sin crear ni entrenar nada.
4. Para una corrida nueva, rechaza un directorio de modelo no vacío; después crea las carpetas y guarda `config.json` con configuración, commit, versiones, memoria y checkpoint de origen.
5. Construye el entorno vectorizado.
6. Si se usa memoria optimizada, pasa `handle_timeout_termination=False`, porque SB3 no permite combinar esa opción con el manejo de truncamientos por timeout. Este entorno no tiene límite temporal por defecto.
7. En una corrida nueva instancia `DQN` con `CnnPolicy`, todos los hiperparámetros y `tensorboard_log`. En una reanudación carga el checkpoint y opcionalmente su replay buffer.
8. Calcula `remaining = total_steps - model.num_timesteps`. Si no se guardó el replay buffer, mueve `learning_starts` hacia adelante para evitar muestrear un buffer vacío.
9. Combina `CheckpointCallback` y `MarioDQNCallback`. El primero guarda archivos `.zip` y, si se solicita, el replay buffer; el segundo registra métricas.
10. Llama a `model.learn` con `reset_num_timesteps=False` al reanudar, preservando la numeración global.
11. Guarda el modelo final como `dqn_s<seed>_final.zip`.

## Archivos producidos

Para una corrida llamada `dqn_s1` se esperan:

```text
models/dqn/dqn_s1/
├── config.json
├── dqn_s1_100000_steps.zip
└── dqn_s1_final.zip

runs/dqn/dqn_s1/
└── episodes.csv
```

Además, SB3 crea eventos de TensorBoard bajo la raíz de logs. Los checkpoints solo aparecen cuando `checkpoint_freq` se alcanza; con un total menor que esa frecuencia puede existir únicamente el modelo final.
