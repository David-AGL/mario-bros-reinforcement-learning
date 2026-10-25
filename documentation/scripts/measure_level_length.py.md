# `scripts/measure_level_length.py`

Herramienta interactiva para medir `level_length`, definido como el valor de `x_pos` cuando Mario toma la bandera. El valor se guarda en `levels.json` y después permite calcular `progress` durante la evaluación.

## Preparación

Importa `argparse`, `json` y `Path`. `ROOT` apunta a la raíz del repositorio y `LEVELS_JSON` a su archivo de niveles.

El script usa `gym_super_mario_bros` y `nes_py.app.play_human`, por lo que necesita una ventana gráfica. En Windows puede requerir WSL2 con WSLg; en Linux necesita un display funcional.

## `save_level_length(level, x_pos, path)`

1. Lee el JSON completo.
2. Recorre todas las secciones y busca el nivel.
3. Convierte `x_pos` a entero y actualiza `level_length`.
4. Escribe el JSON indentado y conserva caracteres Unicode.
5. Devuelve `True` si encontró y actualizó el nivel; devuelve `False` si el nivel no aparece.

La función actualiza el archivo de una sola vez después de localizar el nivel, conservando el resto de secciones y niveles.

## Clase `FlagWatcher`

Es un callback invocable que observa la información devuelta por `env.step`.

- `level` identifica el nivel que se está jugando.
- `write` controla si debe modificar `levels.json`.
- `max_x` conserva la mayor posición observada.
- `done_flag` impide procesar dos veces la bandera.

`__call__(info)` ignora entradas que no sean diccionarios, actualiza `max_x` y reacciona al primer `info["flag_get"]` verdadero. Imprime el `x_pos` de la bandera, opcionalmente llama a `save_level_length`, y avisa que se puede cerrar la ventana.

## `main()`

Define:

- `--level`, por defecto `SuperMarioBros-1-1-v0`;
- `--no-write`, que permite medir sin modificar el JSON.

Luego crea el entorno sin wrappers modernos usando `.unwrapped`, construye el watcher e intercepta temporalmente `env.step`. Esta intercepción es necesaria porque `play_human` descarta el cuarto valor (`info`) del `step` antiguo de `nes-py`; el wrapper local conserva todos los valores y pasa `info` al watcher.

`try/finally` garantiza que el emulador se cierre aunque el usuario cierre la ventana o ocurra una excepción. Si nunca se toma la bandera, solo se imprime el máximo observado y no se escribe ningún valor.

## Consideración sobre niveles con tuberías

En niveles con cambio de área, como `1-2`, `x_pos` puede no crecer de manera continua. Por eso una longitud guardada allí puede no ser adecuada para interpretar `progress`; conviene usar únicamente tasa de completado si la métrica no es fiable.

