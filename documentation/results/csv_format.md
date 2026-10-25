# Resultados CSV

Los CSV en `results/eval/` y `results/val/` contienen una fila por episodio. Sus columnas son:

- `algo`, `seed`, `level`, `episode`: identidad de la corrida.
- `completed`: `1` si se detectó `flag_get`.
- `max_x`, `end_x`, `death_x`: progreso espacial máximo, posición final y posición donde terminó una partida no completada.
- `steps`, `reward`, `elapsed_seconds`: coste y resultado del episodio.
- `prediction_mode`: `greedy`, `epsilon_greedy_0.05` o `uniform_random`.
- `eval_seed`, `checkpoint_steps`, `noop_max`, `noop_used`: condiciones de evaluación.
- `level_length`, `progress`: longitud conocida y proporción `max_x / level_length`.
- `truncated`: indica si terminó por el límite de evaluación.
- `git_commit`: versión del código evaluado.

Los campos de progreso pueden estar vacíos cuando `levels.json` todavía no contiene una longitud válida.

