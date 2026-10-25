# `results/eval/dqn_1-1_greedy_seeds100-104.json`

Resultado JSON de cinco episodios de evaluación del checkpoint DQN con política greedy en `SuperMarioBros-1-1-v0`, usando semillas `100–104`.

## `meta`

Registra los argumentos usados, `env-v1`, el commit `76c1262`, el modo `greedy` y el checkpoint de 50000 pasos. `record=true` indica que también se generó un video por semilla.

## `summary`

- `episodes=5`
- `completion_rate=0.0`
- `max_x_mean=312.0`
- `max_x_std=0.0`
- `progress_mean` y `progress_std` son `null` porque `level_length` estaba vacío.

## `episodes`

Cada objeto representa una semilla y contiene recompensa, pasos, no-ops, posición final/máxima, completado, truncamiento y metadatos de reproducibilidad. En esta corrida todas las partidas terminaron sin tomar la bandera y alcanzaron `max_x=312`.

