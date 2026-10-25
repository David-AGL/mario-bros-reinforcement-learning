# `evaluate.py`

Script principal para evaluar agentes de Mario Bros bajo un protocolo reproducible. Permite comparar un modelo DQN, un modelo PPO y una política aleatoria usando exactamente el mismo entorno, semillas, límite de pasos y métricas.

## Objetivo y protocolo

El script ejecuta un episodio por cada semilla indicada en `--seeds`. Al reiniciar el entorno, `env-v1` aplica entre cero y `noop_max` acciones `NOOP`; como el entorno se reinicia con la semilla de evaluación, esa aleatoriedad es reproducible.

Hay tres modos de comportamiento:

- **DQN estocástico:** usa epsilon-greedy con un `epsilon` fijo definido por `--epsilon`. El `exploration_rate` que el modelo pudo conservar del entrenamiento no se utiliza.
- **PPO estocástico:** llama a la política para muestrear una acción.
- **Determinista:** DQN usa siempre la acción greedy y PPO usa la acción determinista.
- **Random:** escoge uniformemente cualquier acción y no necesita un archivo de modelo.

La métrica `progress` se calcula como `max_x / level_length` si `levels.json` ya contiene la longitud del nivel. `completed` se basa en `info["flag_get"]`, no en alcanzar el límite de pasos.

## Importaciones y constantes

- `argparse` define la interfaz de consola.
- `csv` y `json` escriben los resultados en dos formatos.
- `re` extrae la semilla de entrenamiento desde el nombre del checkpoint.
- `subprocess` obtiene el commit actual de Git.
- `cv2` procesa frames y crea videos.
- `numpy` genera acciones aleatorias y calcula promedios/desviaciones.
- `make_env` crea el entorno común usado por todas las políticas.
- `CSV_FIELDS` fija el orden y el contrato de las columnas de salida.
- `VIDEO_FPS = 15` corresponde a los 60 FPS del NES divididos por el `FRAME_SKIP=4` del entorno.

`ROOT` es la carpeta del repositorio, no la carpeta desde la que se ejecuta el comando. Esto permite encontrar `levels.json` y guardar videos de forma consistente.

## `parse_seeds(text)`

Convierte la notación de consola en una lista de enteros:

- `"200-209"` se convierte en `[200, 201, ..., 209]`.
- `"1,2,5"` se convierte en `[1, 2, 5]`.

Separa primero por comas y luego interpreta cada segmento con guion como un rango inclusivo. Los espacios alrededor de cada segmento se eliminan.

## `git_commit()`

Ejecuta `git rev-parse --short HEAD` desde la raíz y devuelve el commit corto que produjo la evaluación. Si Git no está disponible o la carpeta no es un repositorio, devuelve `"unknown"` para que la evaluación pueda continuar y dejar constancia de la ausencia de metadatos.

## `load_level_length(level, path)`

Lee `levels.json`, recorre sus secciones (`train`, `unseen`, `optional`) y busca el identificador del nivel. Devuelve su `level_length` o `None` si el archivo no existe, no se puede leer o el nivel aún no fue medido. La ausencia no detiene la evaluación: únicamente deja `progress` vacío y muestra una advertencia.

## Clase `Policy`

Unifica la interfaz de DQN, PPO y random para que `run_episode` no tenga que conocer los detalles de cada algoritmo.

### Constructor

Recibe `algo`, el modelo opcional, `mode`, `epsilon` y número de acciones. Valida que el algoritmo sea `dqn`, `ppo` o `random`, y que el modo sea `stochastic` o `deterministic`. Inicializa un generador NumPy separado del generador del entorno.

### Propiedad `prediction_mode`

Devuelve el nombre que se guarda en los resultados:

- `uniform_random` para random.
- `epsilon_greedy_0.05` para DQN estocástico.
- `greedy` para DQN determinista.
- `sample` o `deterministic` para PPO.

Esto hace explícita la fuente de aleatoriedad usada en cada experimento.

### `reset(seed)`

Reinicia el generador de la política con una semilla compuesta `[seed, 1]`. La política y los no-ops del entorno tienen así fuentes separadas: cambiar la secuencia de acciones no cambia el no-op inicial.

Para PPO estocástico también llama a `torch.manual_seed(seed)` para hacer reproducible el muestreo de la política.

### `act(obs)`

- **Random:** genera un entero uniforme entre cero y `n_actions - 1`.
- **DQN estocástico:** con probabilidad `epsilon` genera una acción aleatoria; en otro caso llama a `model.predict(obs, deterministic=True)`.
- **DQN determinista:** siempre usa `model.predict(..., deterministic=True)`.
- **PPO:** usa `deterministic=True` solo en modo determinista; en modo estocástico permite que SB3 muestree la política.

El resultado siempre se convierte a `int`, que es el tipo requerido por el espacio discreto del entorno.

## `load_policy(...)`

Devuelve una tupla `(policy, checkpoint_steps)`.

- Para `random`, crea una `Policy` sin cargar modelo y devuelve cero pasos.
- Para DQN importa `stable_baselines3.DQN` y carga el `.zip` con `DQN.load`.
- Para PPO importa `stable_baselines3.PPO` y carga el `.zip` con `PPO.load`.
- Usa `model.action_space.n` para conservar el número real de acciones del checkpoint.
- Devuelve `model.num_timesteps` como cantidad de pasos del checkpoint.

Si se solicita DQN o PPO sin `--model`, termina con un mensaje explícito.

## `run_episode(env, policy, eval_seed, max_steps, record_path)`

Ejecuta un episodio completo y devuelve un diccionario de métricas.

1. Reinicia el entorno con `eval_seed` y obtiene `noop_used`.
2. Reinicia la política con la misma semilla.
3. Inicializa recompensa, posición máxima, posición final, contador de pasos y banderas.
4. Si se solicitó grabación, obtiene el frame inicial con `env.render()`, lo escala al doble con vecino más cercano y crea un `cv2.VideoWriter` en el primer frame válido.
5. En cada iteración pide una acción, ejecuta `env.step`, acumula recompensa y actualiza `end_x`, `max_x` y `completed`.
6. Escribe un frame después de cada acción cuando se está grabando.
7. Finaliza ante `terminated`, `truncated` o al alcanzar `max_steps`. El último caso se marca explícitamente como truncamiento de evaluación.
8. Libera el escritor de video.

`death_x` contiene `end_x` solo si no se completó el nivel; en una victoria queda vacío. La función no escribe CSV ni conoce argumentos de consola, por lo que puede reutilizarse desde una interfaz futura.

## `summarize(rows)`

Calcula:

- número de episodios;
- tasa de completado;
- media y desviación estándar de `max_x`;
- media y desviación estándar de `progress` solo cuando todos los episodios tienen longitud de nivel.

Si falta `level_length` en alguna fila, las métricas de progreso se dejan como `None` para evitar promedios parciales engañosos.

## `main(argv=None)`

1. Define los argumentos: algoritmo, modelo, nivel, semillas, modo, epsilon, no-ops, límite de pasos, dispositivo, carpeta de salida, etiqueta y grabación.
2. Parsea las semillas y carga la política.
3. Obtiene la semilla de entrenamiento desde `--train-seed` o del patrón `_s<number>` del nombre del modelo.
4. Busca la longitud del nivel y avisa si no está disponible.
5. Crea un único `MarioEnv`; activa `render_mode="rgb_array"` solo si se solicitaron videos.
6. Construye una etiqueta automática como `dqn_1-1_epsilon_greedy_0.05_seeds100-104`, salvo que se proporcione `--tag`.
7. Recorre las semillas, ejecuta `run_episode`, añade metadatos y muestra una línea de progreso por episodio.
8. Cierra el entorno.
9. Crea `--out`, escribe `<tag>.csv` con `CSV_FIELDS` y `<tag>.json` con argumentos, commit, versión del entorno, resumen y episodios.

## Campos de salida

Las filas incluyen algoritmo, semilla de entrenamiento y evaluación, nivel, episodio, completado, `max_x`, pasos, recompensa, tiempo, modo de predicción, checkpoint, no-ops, longitud, progreso, posición de muerte/final, truncamiento y commit.

## Ejemplos

```bash
python evaluate.py --algo random --level SuperMarioBros-1-1-v0 --seeds 100-104
python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --seeds 100-104 --out results/val
python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --mode deterministic --seeds 100-104 --record
```

