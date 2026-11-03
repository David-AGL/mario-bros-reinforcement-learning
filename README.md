# 🍄 Super Mario Bros — Aprendizaje por Refuerzo (DQN + PPO)

> Agentes entrenados con Deep Q-Network y Proximal Policy Optimization para completar el nivel 1-1 de Super Mario Bros, con una interfaz de evaluación interactiva desplegada en Streamlit Cloud.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mario-bros-reinforcement-learning-ptlu26zjorgag6xbfxsc6z.streamlit.app)

🔗 **Demo en vivo:** [https://mario-bros-reinforcement-learning-ptlu26zjorgag6xbfxsc6z.streamlit.app](https://mario-bros-reinforcement-learning-aoejjycte8h8a53uwlqeq.streamlit.app)

---

## 🏆 Resultados de los Agentes

| # | Agente | Algoritmo | Pasos | Progreso (Mejor record) | Récord Pico | Bandera      |
|---|--------|-----------|-------|-------------------------|------------|--------------|
| 1 | **PPO Campeón** | PPO | 3 M | **100.0%**              | 3161 px (100%) | ✅ 20/20 eps |
| 2 | DQN Advanced / Fine-Tuning | DQN | 450 k | 78%                     | 2463 px (78%) | No           |
| 3 | DQN CNN Base | DQN | 900 k | 48.2%                   | 1523 px (48%) | No           |
| 4 | Agente Aleatorio (Línea Base) | Random | — | 25.1%                   | — | No           |

---

## 📁 Estructura del Proyecto

```
mario-bros-reinforcement-learning/
├── app/
│   └── main.py              # Interfaz Streamlit (Panel de Control)
├── dqn/
│   └── train.py             # Entrenamiento DQN local (CLI)
├── models/
│   ├── ppo/                 # Modelo PPO entrenado (ppo_s1_3m.zip)
│   └── dqn/                 # Modelos DQN (archivos .zip, excluidos de git)
├── notebooks/
│   ├── colab_train.ipynb        # Entrenamiento DQN en Google Colab
│   └── transfer_learning_mario.ipynb  # Transfer Learning con NatureCNN preentrenado
├── results/
│   └── val/                 # CSVs de validación por agente y checkpoint
├── videos/                  # Videos pregrabados de las mejores partidas (.mp4)
├── env.py                   # Wrapper del entorno (env-v2)
├── evaluate.py              # Script de evaluación y grabación de episodios
├── levels.json              # Longitudes de niveles por nivel_id
├── packages.txt             # Dependencias del sistema (ffmpeg) para Streamlit Cloud
└── requirements.txt         # Dependencias Python
```

---

## 🕹️ Entorno (env-v2)

- **Acciones:** `SIMPLE_MOVEMENT` (7 acciones)
- **Observación:** escala de grises · 84×84 · stack de 4 frames → shape `(4, 84, 84)` uint8
- **Frame skip:** 4
- **Recompensa:** avance + tiempo − 15 al morir + bandera +500 + hongo/flor +100 − poder perdido −10
- **Nivel fijo de evaluación:** `SuperMarioBros-1-1-v0` (longitud: 3 161 px)

Ver [`env.py`](env.py) para todos los wrappers.

---

## 🚀 Instalación local

```bash
# 1. Instalar PyTorch según tu CUDA: https://pytorch.org
# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar entorno
python scripts/smoke_test_env.py   # debe terminar en "SMOKE TEST OK"
```

> **WSL Ubuntu recomendado** para la simulación en vivo (`nes-py` no compila fácilmente en Windows nativo).

---

## 🖥️ Interfaz Streamlit (local)

```bash
# Desde el entorno virtual con nes-py instalado:
streamlit run app/main.py
```

La interfaz incluye:
- **Panel de Métricas y Video** — métricas por agente + video pregrabado de su mejor partida
- **Modo en Vivo** — ejecuta un episodio real en el emulador NES y graba el video
- **Comparativa de Modelos** — tabla y gráficas comparando los 4 agentes

---

## 📊 Evaluación

```bash
# Línea base aleatoria
python evaluate.py --algo random --seeds 100-104

# DQN
python evaluate.py --algo dqn \
  --model models/dqn/dqn_v2_explore05_s1/dqn_v2_explore05_900k.zip \
  --seeds 100-104 --out results/val

# PPO (modo determinístico)
python evaluate.py --algo ppo \
  --model models/ppo/ppo_s1_3m.zip \
  --mode deterministic --seeds 100-104 --record
```

**Protocolo:**
- 1 episodio por semilla (seeds 100–104, 20 eps para PPO)
- No-ops aleatorios 0–30 al reiniciar
- DQN: epsilon-greedy fijo ε=0.05
- PPO: muestreo estocástico de la política
- Salida: `<out>/<tag>.csv` con columnas `algo, seed, level, episode, max_x, progress, reward, completed, death_x, steps`

---

## 🧠 Flujo de Entrenamiento

| Flujo | Descripción |
|-------|-------------|
| **Colab DQN** | `notebooks/colab_train.ipynb` — entrena con checkpoints sincronizados en Google Drive |
| **Transfer Learning** | `notebooks/transfer_learning_mario.ipynb` — inicializa `NatureCNN` con pesos PPO de Hugging Face para acelerar la convergencia DQN |
| **Local CLI** | `dqn/train.py --models-root ... --logs-root ...` |

---

## 📦 Dependencias del Sistema (Streamlit Cloud)

El archivo [`packages.txt`](packages.txt) instala `ffmpeg` en el servidor de Streamlit Cloud, necesario para la conversión de video H.264.

---

## 📝 Notas

- Los archivos `.zip` de modelos están excluidos de git (demasiado grandes). Se descargan automáticamente desde Google Drive al hacer clic en **"Ejecutar Episodio en Vivo"** en la app.
- Los videos en `videos/live_*.mp4` y `videos/*_raw.mp4` también están excluidos.
- Los 4 videos pregrabados de los agentes **sí están incluidos** en el repositorio.
