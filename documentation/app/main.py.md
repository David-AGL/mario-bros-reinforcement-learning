# `app/main.py`

Interfaz de usuario en **Streamlit** para la evaluación y playtesting de Mario Bros con Aprendizaje por Refuerzo (Fase 1 del plan de trabajo).

## Componentes implementados

1. **Selector de Agente:**
   - `DQN Mejorado (900k pasos - explore 0.5)` (el mejor modelo).
   - `DQN Inicial (50k pasos)` (línea base DQN).
   - `Agente Aleatorio` (línea base sin entrenamiento).
   - Checkpoints intermedios opcionales.

2. **Selector de Nivel:**
   - `Nivel 1-1 (visto)`
   - `Nivel 2-1 (no visto)`
   - `Nivel 3-1 (no visto)`

3. **Visualizador de Video:**
   - Detección y reproducción de archivos `.mp4` en `videos/` o descargas.
   - Componente de subida de archivos (drag & drop) para reproducir cualquier video de partida.

4. **Modo en Vivo (run_episode):**
   - Botón interactivo para ejecutar un episodio en tiempo real reutilizando `run_episode` de `evaluate.py`.
   - Manejo de excepciones elegante si las librerías de emulación NES no están presentes en el entorno anfitrión.

5. **Panel de Métricas:**
   - **Progreso:** $\frac{\max x}{\text{level\_length}}$ (%) con barra de progreso visual.
   - **Coordenada Máxima:** $\max x$ en píxeles.
   - **Recompensa Acumulada:** `reward`.
   - **Bandera:** Estado de completado (`completed`).
   - **Punto de Muerte:** `death_x`.
   - Tabla detallada por semilla de validación (100–104).

## Cómo ejecutar

Desde PowerShell en Windows o terminal de Linux:
```bash
streamlit run app/main.py
```
Abre la aplicación en el navegador en `http://localhost:8501`.
