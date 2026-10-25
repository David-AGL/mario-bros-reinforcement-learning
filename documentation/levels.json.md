# `levels.json`

Catálogo de niveles y metadatos de longitud:

- `train`: niveles usados para entrenamiento; actualmente `SuperMarioBros-1-1-v0`.
- `unseen`: niveles reservados para evaluación fuera de distribución; actualmente `2-1` y `3-1`.
- `optional`: nivel opcional `1-2`.
- `level_length`: posición `x_pos` al alcanzar la bandera. Actualmente `1-1` tiene `3161`, `2-1` y `3-1` tienen `3193`, y `1-2` tiene `3161`.

El archivo separa explícitamente el nivel de entrenamiento de los niveles no vistos y deja un lugar para comparar cuánto avanza el agente.

## Uso durante la evaluación

`evaluate.py` busca el nivel en todas las secciones y usa este valor como denominador:

```text
progress = max_x / level_length
```

Si se cambia una longitud, las evaluaciones posteriores serán comparables entre sí, pero no necesariamente con CSV/JSON generados antes del cambio. El valor de `1-2` debe interpretarse con cuidado porque los cambios de área pueden hacer que `x_pos` no represente una distancia continua.
