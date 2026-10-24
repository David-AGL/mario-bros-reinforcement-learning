# DQN — hiperparámetros y memoria

| Parámetro | Valor | Nota |
|---|---|---|
| Política | CnnPolicy (NatureCNN) | SB3 normaliza /255 internamente |
| Entornos | 1 | |
| buffer_size | 100 000 | ver memoria |
| learning_starts | 20 000 | |
| learning_rate | 1e-4 | |
| batch_size | 32 | |
| gamma | 0.99 | |
| train_freq / gradient_steps | 4 / 1 | |
| target_update_interval | 10 000 | |
| exploration_fraction | 0.2 | de `total_steps` (cambia con el presupuesto) |
| exploration_final_eps | 0.05 | |
| max_grad_norm | 10 | default de SB3 |

## Memoria del replay buffer
Observación = 4·84·84 = 28 224 B (uint8).

| buffer_size | normal (obs + next_obs) | `--optimize-memory-usage` |
|---|---|---|
| 100 000 | 5.64 GB | 2.82 GB |
| 50 000 | 2.82 GB | 1.41 GB |

Con 16 GB de RAM (15.7 usables) conviene medir el RSS real (`system/rss_gb` en TensorBoard)
antes de entrenar DQN y PPO a la vez.

## Decisiones por acordar
- `noop_max` en entrenamiento (por defecto 30; la evaluación ya usa 0–30).
- Presupuesto de pasos (igual para DQN y PPO) tras medir FPS.
- Guardar o no el replay buffer en los checkpoints (por defecto no).
