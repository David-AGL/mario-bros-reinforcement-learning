# `levels.json`

Catálogo de niveles y metadatos de longitud:

- `train`: niveles usados para entrenamiento; actualmente `SuperMarioBros-1-1-v0`.
- `unseen`: niveles reservados para evaluación fuera de distribución; actualmente `2-1` y `3-1`.
- `optional`: nivel opcional `1-2`.
- `level_length`: posición `x_pos` al alcanzar la bandera. Es `null` mientras no se haya medido.

El archivo separa explícitamente el nivel de entrenamiento de los niveles no vistos y deja un lugar para comparar cuánto avanza el agente.

