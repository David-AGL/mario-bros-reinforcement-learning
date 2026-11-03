# `notebooks/model_development.ipynb`

Notebook de análisis del entregable final para comparar DQN y PPO y analizar el comportamiento de los agentes como jugadores automáticos. Consolida los experimentos ejecutados por los lanzadores de Colab/Kaggle y el experimento de Transfer Learning, sin reemplazar esos notebooks.

## Celdas conceptuales

1. **Contexto:** define el objetivo de playtesting automático, los usuarios y la decisión de acción que apoya el agente.
2. **Datos y entorno:** explica que las trayectorias se generan con `gym-super-mario-bros`/`nes-py`, el preprocesamiento visual de `env-v2`, las recompensas y las variables que no entran a la política.
3. **Algoritmos y decisiones:** presenta DQN como foco, PPO como baseline y justifica replay buffer, red objetivo, epsilon-greedy, GAE y clipping.
4. **Entrenamiento:** documenta el paso a paso común y distingue Colab/Kaggle como lanzadores de infraestructura de Transfer Learning como variante experimental.
5. **Protocolo:** establece un episodio por semilla, validación `100–104`, evaluación final `200–209`, no-ops `0–30`, epsilon DQN fijo `0.05` y resultado determinista secundario.
6. **Análisis exploratorio:** carga todos los CSV de `results/` y analiza recompensas, duración, progreso, completitud y posiciones de muerte, que sustituyen el EDA de un dataset estático.
7. **Resultados e insights:** genera un resumen agrupado y una gráfica de progreso, y deja criterios explícitos para comparar algoritmos, niveles vistos/no vistos y entrenamiento desde cero/transfer learning.
8. **Generalización y despliegue:** documenta limitaciones, evaluación en niveles no vistos y el uso de Streamlit para auditar agentes.

El notebook no genera datos artificiales ni escribe métricas a mano: consume los CSV producidos por `evaluate.py`. La función de carga recursiva permite incorporar nuevas corridas sin cambiar el análisis. Si no existen resultados, muestra avisos en lugar de fabricar conclusiones.
