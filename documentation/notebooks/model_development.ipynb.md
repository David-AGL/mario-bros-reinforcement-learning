# `notebooks/model_development.ipynb`

Notebook de desarrollo del entregable final para comparar DQN y PPO y analizar el comportamiento de los agentes como jugadores automáticos.

## Celdas conceptuales

1. **Problema:** define el objetivo de playtesting automático: medir progreso, completado y zonas donde muere el agente.
2. **Entorno:** fija `env-v1`, las transformaciones visuales, los niveles de entrenamiento/no vistos y el uso de `levels.json`.
3. **Algoritmos:** presenta DQN como foco, con replay buffer, red objetivo y epsilon-greedy; presenta PPO como baseline con política estocástica, GAE y clipping.
4. **Protocolo:** establece un episodio por semilla, semillas de validación `100–104`, evaluación final `200–209`, no-ops `0–30`, epsilon DQN fijo `0.05` y resultado determinista secundario.
5. **Carga de datos:** importa pandas, glob y Path; `RESULTS` apunta a `results/`; `load(pattern)` busca todos los CSV y los concatena en un único DataFrame. Si no encuentra archivos, devuelve un DataFrame vacío.
6. **Resultados:** deja el espacio para generar tablas y curvas a partir de los datos crudos, separando niveles vistos y no vistos.
7. **Tabla comparativa:** agrupa por algoritmo, nivel y modo de predicción; calcula cantidad de episodios, media de `max_x` y media de `completed`, y muestra la tabla si hay datos.
8. **Generalización y limitaciones:** indica que los niveles no vistos deben evaluarse al final y que no se debe afirmar que el agente completó un nivel si únicamente se observó progreso.

El notebook no genera datos artificiales ni escribe métricas a mano: consume los CSV producidos por `evaluate.py`. La función de carga recursiva permite incorporar nuevas corridas sin cambiar el análisis.

