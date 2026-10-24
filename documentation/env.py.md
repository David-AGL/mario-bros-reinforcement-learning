# `env.py`

Implementa el entorno Gymnasium `MarioEnv`, un adaptador entre `gym-super-mario-bros`/`nes-py` y la API que utiliza Stable-Baselines3 2.x. El archivo es el contrato central del proyecto: cualquier modelo entrenado depende exactamente de sus acciones, forma de observación, preprocesamiento y manejo de episodios.

## Importaciones

- `deque` mantiene los últimos frames sin tener que administrar manualmente una lista de tamaño fijo.
- `cv2` convierte y redimensiona las imágenes.
- `gymnasium` proporciona la clase base `Env` y la API moderna `reset`/`step`.
- `numpy` representa los frames y las observaciones.
- `spaces` define los espacios de acciones y observaciones.
- `gym_super_mario_bros` crea el juego de NES.
- `SIMPLE_MOVEMENT` define las siete acciones discretas disponibles.
- `JoypadSpace` traduce el índice de acción a los botones del control.

## Constantes y contrato

- `ENV_VERSION = "env-v1"` identifica el preprocesamiento.
- `FRAME_SKIP = 4` repite la acción durante cuatro frames y acumula sus recompensas.
- `FRAME_SIZE = 84` fija el ancho y alto de cada frame.
- `N_STACK = 4` mantiene cuatro frames para representar movimiento.
- `NOOP_ACTION = 0` corresponde a `NOOP` en `SIMPLE_MOVEMENT`.
- `action_space` es `Discrete(7)`.
- `observation_space` es `Box(0, 255, (4, 84, 84), uint8)`.

No se recorta ni escala la recompensa original del juego.

## Función `level_id`

`level_id(world, stage, version="v0")` recibe mundo, nivel y versión y devuelve el identificador que espera `gym_super_mario_bros.make`. Por ejemplo, `level_id(1, 1)` genera `SuperMarioBros-1-1-v0`. La función solo formatea texto; no verifica que el nivel exista, por lo que esa comprobación ocurre al crear el entorno.

## Clase `MarioEnv`

Hereda de `gym.Env` y declara `metadata = {"render_modes": ["rgb_array"]}`. Esto indica que la única modalidad de render soportada explícitamente es devolver una imagen RGB en memoria.

### Constructor `__init__`

1. Guarda `env_id`, `noop_max`, `max_episode_steps` y `render_mode`. `noop_max` se convierte a entero para evitar recibir valores numéricos ambiguos.
2. Llama a `super().__init__()` para inicializar el entorno Gymnasium.
3. Crea el juego con `gym_super_mario_bros.make(env_id).unwrapped`. `.unwrapped` evita wrappers de Gym 0.26 que esperan la API nueva, mientras que `nes-py` todavía devuelve la API antigua.
4. Envuelve el juego con `JoypadSpace(base, SIMPLE_MOVEMENT)`, de modo que el agente envía un entero y el wrapper pulsa la combinación de botones correspondiente.
5. Verifica con un `assert` que la acción cero sea realmente `NOOP`; esto protege la lógica de reinicio contra un cambio accidental en `SIMPLE_MOVEMENT`.
6. Define `action_space` y `observation_space`.
7. Inicializa `_frames` como una cola de máximo cuatro elementos, `_steps` como contador del episodio y `_last_raw` para guardar el último frame RGB.

### `_process(rgb)`

Es un método estático porque no necesita el estado de la instancia. Recibe un frame RGB, lo convierte a escala de grises con `cv2.COLOR_RGB2GRAY` y lo reduce a `84x84` con interpolación `INTER_AREA`. Devuelve un arreglo de un canal cuyo tipo normalmente es `uint8`. Esta transformación reduce memoria y hace que el modelo trabaje con una entrada fija.

### `_obs()`

Apila los elementos de `_frames` en el eje cero mediante `np.stack`. El resultado tiene el orden `(frames, height, width)`, es decir `(4, 84, 84)`. No normaliza a `[0, 1]`; Stable-Baselines3 normaliza internamente la entrada de `CnnPolicy`.

### `reset(seed=None, options=None)`

1. Llama a `super().reset(seed=seed)`. Además de cumplir la API Gymnasium, esto crea/siembra `self.np_random`, que luego se usa para elegir los no-ops reproducibles.
2. Reinicia el emulador y guarda el frame RGB inicial en `_last_raw`; también reinicia `_steps`.
3. Si `noop_max > 0`, escoge uniformemente un entero entre `0` y `noop_max` inclusive con `self.np_random.integers`.
4. Ejecuta esa cantidad de veces la acción `NOOP_ACTION`. Aunque esas transiciones avanzan el juego, sus recompensas no se devuelven al agente porque ocurren dentro del reinicio.
5. Si el juego termina durante los no-ops, reinicia sin no-ops y registra `noop_used=0`; esto evita entregar al agente un episodio muerto.
6. Procesa el frame final y lo repite cuatro veces para llenar la cola inicial. Así, la primera observación siempre tiene la forma correcta.
7. Copia `info`, añade `noop_used` y `env_id`, y devuelve `(observación, info)`.

El parámetro `options` se acepta por compatibilidad con Gymnasium, pero actualmente no modifica el reinicio.

### `step(action)`

1. Inicializa la recompensa acumulada, la bandera `terminated` y las variables temporales.
2. Repite la acción hasta `FRAME_SKIP` veces. Cada retorno antiguo de `nes-py` contiene `raw, reward, done, info`.
3. Suma cada recompensa en `total_reward`. Si `done` es verdadero, marca `terminated` y corta el bucle antes de cuatro frames.
4. Guarda el último frame RGB, lo procesa y añade un único frame a `_frames`; la cola elimina automáticamente el más antiguo.
5. Incrementa `_steps`.
6. Calcula `truncated` únicamente cuando el juego no terminó y se alcanzó `max_episode_steps`. De esta forma distingue una derrota/victoria del límite artificial impuesto por el agente.
7. Devuelve la observación apilada, la recompensa total, ambas banderas y una copia del último `info`.

La acción se convierte a `int` antes de enviarse al emulador, lo que permite usar valores NumPy provenientes de un espacio discreto.

### `render`, `close` y `make_env`

- `render` devuelve `_last_raw` solo si se solicitó `render_mode="rgb_array"`; en cualquier otro caso devuelve `None`.
- `close` delega en `_nes.close()` para liberar el emulador.
- `make_env` expone una fábrica conveniente y reenvía argumentos adicionales al constructor. Es la función que usan los scripts de entrenamiento y prueba.

## Decisiones de compatibilidad

El entorno mezcla deliberadamente un backend antiguo (`nes-py`) con la interfaz moderna de Gymnasium. Por eso no se usa directamente el wrapper estándar de Gymnasium y se traducen manualmente los cuatro valores antiguos de `step` a los cinco valores modernos (`terminated` y `truncated` separados). Cambiar esta capa puede invalidar la compatibilidad con los checkpoints existentes.
