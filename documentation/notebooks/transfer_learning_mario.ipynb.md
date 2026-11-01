# `notebooks/transfer_learning_mario.ipynb`

Notebook interactivo para entrenar un agente **DQN** utilizando **Transfer Learning**, acelerando drásticamente el aprendizaje mediante la reutilización de representaciones visuales preentrenadas provenientes de un modelo **PPO** que ya superó el nivel 1-1.

---

## Fundamento teórico y motivación

Entrenar un agente DQN puramente desde cero en *Super Mario Bros* demanda millones de pasos de interacción debido al retraso temporal en las recompensas y la alta dimensionalidad visual del espacio de estados `(4, 84, 84)`.

La red convolucional `NatureCNN` (3 capas convolucionales) aprende a extraer características espaciales universales del videojuego:
- Detección de bordes, plataformas y suelo.
- Reconocimiento de tuberías, bloques con interrogación y ladrillos.
- Localización de enemigos (Goombas, Koopas) y potenciadores (hongos, flores).

Dado que la representación visual es idéntica independientemente del algoritmo de aprendizaje por refuerzo, los pesos de `features_extractor` del modelo PPO preentrenado se extraen e inyectan directamente en la red Q del modelo DQN (`q_net` y `q_net_target`), inicializando únicamente las capas densas finales (*fully connected*) desde cero.

---

## Flujo de celdas del Notebook

### 1. Entorno de ejecución en Google Colab
- Comprobación de disponibilidad de GPU y CUDA con `nvidia-smi` y `torch.cuda`.
- Montaje de Google Drive para almacenamiento persistente de checkpoints y logs.
- Clonación o actualización del repositorio del proyecto.
- Instalación de dependencias fijadas (`numpy<2`, `gymnasium`, `stable-baselines3`, `huggingface_hub`, etc.) desinstalando paquetes incompatibles de Colab.
- Ejecución del smoke test (`scripts/smoke_test_env.py`) para verificar la sanidad del entorno `env-v2`.

### 2. Descarga del modelo preentrenado (Paso 1)
- Descarga automatizada del checkpoint preentrenado desde Hugging Face Hub (`tsilva/Level1-1_stable-baselines3-ppo_7d9b9131/model.zip`).
- Apertura del archivo `.zip` e inspección del `policy.pth` de PyTorch.

### 3. Inspección y extracción de pesos (Paso 2)
- Filtrado de las capas de la `NatureCNN` ignorando las cabezas del actor y del crítico de PPO.
- Mapeo y verificación de formas de tensores (`cnn.0`, `cnn.2`, `cnn.4`, `linear.0`).

### 4. Creación del modelo DQN e inyección de pesos (Paso 3)
- Instanciación de `DQN("CnnPolicy", ...)` con el contrato del entorno `env-v2` (`SIMPLE_MOVEMENT`, 7 acciones).
- Carga de los pesos visuales mediante `load_state_dict(..., strict=False)` tanto en la política activa como en la red objetivo (`q_net_target`).
- Congelación controlada de la CNN (`requires_grad = False`) durante la fase inicial para realizar *feature extraction* antes del ajuste fino.

### 5. Entrenamiento por fases (Paso 4)
- **Fase 1 (0 -> 50,000 pasos):** CNN congelada (`LR = 1e-4`), entrenando rápidamente la cabeza lineal de Q-values.
- **Fase 2 (50,000 -> 500,000 pasos):** Descongelamiento progresivo mediante `UnfreezeCNNCallback` con tasa de aprendizaje reducida (`LR = 1e-5`) para ajuste fino (*fine-tuning*) sin destruir los filtros convolucionales.
- Sincronización periódica en segundo plano hacia Google Drive para tolerar caídas de sesión.

### 6. Evaluación y comparación estructurada (Paso 5)
- Ejecución del protocolo de evaluación estándar de 5 semillas (`seeds 100-104`).
- **Pico de rendimiento del proyecto:**
  - El checkpoint `dqn_adv_450000_steps.zip` (450k pasos) logró el mejor desempeño global de todo el proyecto con un promedio de **1,488.4 px (47.1% del nivel)**.
  - En la semilla individual `104` alcanzó un récord absoluto de **2,466 px (78.0% del nivel)** superando obstáculos avanzados como fosos y tuberías altas finales.

### 7. Grabación y Reproducción de Video Web (Paso 6)
- Grabación reproducible con `EVAL_SEED = 104` y `noop_max = 30` (21 no-ops iniciales) replicando el episodio récord de 412 pasos.
- Conversión automática con `ffmpeg` a códec web `H.264` (`libx264`, `pix_fmt yuv420p`) para visualización fluida en navegadores.
- Renderizado interactivo directamente en la celda con HTML5 y base64, más persistencia automática del video `.mp4` en Google Drive.

---

## Consideraciones de interoperabilidad con el repositorio

- Los checkpoints entrenados utilizan 7 acciones (`SIMPLE_MOVEMENT`) y son **100% compatibles** con `evaluate.py`, `scripts/` y la interfaz de `app/main.py`.
