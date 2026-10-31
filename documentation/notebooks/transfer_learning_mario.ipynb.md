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
- Opción de congelar temporalmente los gradientes de la CNN (`requires_grad = False`) para realizar *feature extraction* puro antes de descongelar.

### 5. Entrenamiento y persistencia (Paso 4)
- Configuración de hiperparámetros adaptados para fine-tuning (menor exploración inicial y `learning_rate` conservador).
- Sincronización periódica en segundo plano hacia Google Drive para tolerar caídas de sesión.
- Guardado de checkpoints intermedios (`100k`, `200k`, etc.) y modelo final.

### 6. Evaluación y comparación estructurada (Paso 5)
- Ejecución del protocolo de evaluación sobre los checkpoints generados (semillas 100–104).
- Extracción de métricas: `max_x`, progreso porcentual, recompensas acumuladas y tasa de banderas conseguidas (`completed`).
- Comparación tabular del agente transfer vs. baselines entrenados desde cero.

### 7. Secciones avanzadas y experimentales
- **Entrenamiento por fases:** Implementación de un `BaseCallback` que descongela gradualmente las capas de la CNN a partir de un umbral de pasos (*fine-tuning progresivo*).
- **Grabación y reproducción de video:** Captura de episodios deterministas del agente campeón con OpenCV y renderizado interactivo en el notebook con HTML5 y base64.
- **Espacio de acciones complejo y Reward Shaping (experimental):** Exploración de un wrapper personalizado (`CustomMarioRewardWrapper`) con penalización por estancamiento horizontal y prueba de `COMPLEX_MOVEMENT` (12 acciones).

---

## Consideraciones de interoperabilidad con el repositorio

- Los checkpoints estándar entrenados con las primeras secciones del notebook utilizan 7 acciones (`SIMPLE_MOVEMENT`) y son **100% compatibles** con `evaluate.py`, `scripts/` y la interfaz de `app/main.py`.
- Si se ejecutan los experimentos opcionales con `COMPLEX_MOVEMENT` (12 acciones), dichos checkpoints requieren registrar explícitamente el espacio de acción modificado para no romper el contrato de evaluación predeterminado del proyecto.
