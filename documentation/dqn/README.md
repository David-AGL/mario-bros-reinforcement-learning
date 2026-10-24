# `dqn/README.md`

Documenta los hiperparámetros iniciales del agente DQN y la memoria requerida por el replay buffer.

## Hiperparámetros

Describe `CnnPolicy`/`NatureCNN`, un solo entorno, `buffer_size=100000`, `learning_starts=20000`, tasa `1e-4`, batch de 32, `gamma=0.99`, frecuencia de entrenamiento 4, actualización del target cada 10000 pasos y decaimiento de exploración hasta `0.05`.

## Memoria

Calcula que una observación ocupa `4 * 84 * 84 = 28224` bytes. El buffer normal conserva observaciones actuales y siguientes; `optimize-memory-usage` reutiliza memoria y aproximadamente reduce a la mitad el consumo.

La advertencia es importante porque un buffer de 100000 puede ocupar varios gigabytes y competir con la memoria necesaria para entrenar.

## Decisiones pendientes

Registra decisiones experimentales todavía abiertas: `noop_max` de entrenamiento, presupuesto de pasos común para DQN y PPO, y si guardar el replay buffer en checkpoints.

