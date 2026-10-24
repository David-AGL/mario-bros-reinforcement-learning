# `scripts/smoke_test_env.py`

Prueba operativa del entorno antes de entrenar.

## Preparación

El script importa `sys`, `time` y `Path`, añade al `sys.path` la raíz del repositorio (`parents[1]`) y luego importa NumPy y las constantes/fábrica de `env.py`. Esto permite ejecutarlo directamente como `python scripts/smoke_test_env.py` sin instalar el repositorio como paquete.

`LEVELS` contiene los cuatro identificadores que deben abrirse correctamente: el nivel de entrenamiento `1-1`, dos niveles no vistos (`2-1` y `3-1`) y el nivel opcional `1-2`.

## `check_obs(obs, env)`

Valida una observación después de cada reinicio y paso:

1. `obs.shape` debe ser `(N_STACK, FRAME_SIZE, FRAME_SIZE)`, que en la configuración actual es `(4, 84, 84)`.
2. `obs.dtype` debe ser `np.uint8`.
3. El mínimo y máximo deben estar en `[0, 255]`.
4. `env.observation_space.contains(obs)` debe aceptar la observación completa.

La función usa `assert`, por lo que una incompatibilidad detiene el smoke test inmediatamente y muestra el valor recibido.

## `run(env_id, noop_max, n_steps=2000, seed=0)`

1. Crea el entorno con `make_env` y reinicia con la semilla indicada.
2. Valida la observación inicial y comprueba que `noop_used` esté entre cero y `noop_max`.
3. Ejecuta `n_steps` acciones aleatorias tomadas de `env.action_space`.
4. Valida cada observación, conserva el mayor `x_pos` presente en `info` y cuenta episodios terminados o truncados.
5. Reinicia el entorno cuando acaba un episodio, sin pasar una nueva semilla.
6. Calcula FPS como pasos del agente divididos por tiempo transcurrido, cierra el entorno y devuelve episodios, máximo `x_pos` y FPS.

El FPS medido corresponde a pasos del agente, no a frames individuales del emulador: cada paso puede ejecutar hasta cuatro frames por `FRAME_SKIP`.

## Bloque principal

Cuando el archivo se ejecuta directamente, recorre todos los niveles con `noop_max=30` y muestra una línea de resultado por nivel. Después crea otro entorno y reinicia tres veces con `seed=200`; exige que los tres valores de `noop_used` sean iguales para verificar reproducibilidad. Finalmente imprime `SMOKE TEST OK`.

Si todo es correcto imprime una línea por nivel y termina con `SMOKE TEST OK`; cualquier incumplimiento produce `AssertionError`.
