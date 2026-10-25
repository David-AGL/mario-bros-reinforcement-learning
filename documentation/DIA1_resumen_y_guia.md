# Día 1 — Resumen de lo hecho y guía para continuar

Proyecto: Mario Bros con aprendizaje por refuerzo (DQN foco, PPO baseline). Plan de 7 días, 3 personas (P1, P2, P3).
Este documento recoge **qué se hizo el día 1, qué comandos se ejecutaron, qué problemas aparecieron** y
**qué debe ejecutar cada integrante para continuar**. Lo que no se pudo verificar se marca como *sin verificar*.

---

## 1. Estado del día 1

| Tarea del plan | Estado | Nota |
|---|---|---|
| Repo, README inicial, `.gitignore` | Hecho | Ver §6: se propone cambiar el `.gitignore` |
| `requirements.txt` coherente y probado | Hecho | Probado en Colab (Python 3.13) y WSL2 Ubuntu (Python 3.12) |
| IDs exactos de niveles | Hecho | Se congela la versión `v0` (ej. `SuperMarioBros-1-1-v0`) |
| `env.py` + smoke test (P1) | Hecho | Pasa en 4 niveles, en Colab y en local |
| Esqueleto de `dqn/train.py`, hiperparámetros, memoria del buffer (P2) | Hecho | Además se corrió un smoke test de 50k pasos (tarea del día 2) |
| Esqueleto de `evaluate.py` y del notebook (P3) | Hecho | `evaluate.py` ya es funcional; notebook = esqueleto |
| `level_length` en JSON (P3) | Hecho, con reserva | El valor de 1-2 es dudoso (§7) |
| Puerta de entrada (smoke test OK antes de entrenar) | Superada | |
| Contratos, formato de resultados, nombres de modelos y **ramas** | Parcial | Formato y nombres definidos; **ramas por acordar** |
| Prueba de reanudación tras caída (Colab) | **Pendiente** | Sin verificar |

---

## 2. Estructura del repositorio

```
mario-bros-reinforcement-learning/
├── env.py                    # Entorno env-v1 (congelado)
├── evaluate.py               # Evaluación reproducible (DQN / PPO / aleatorio)
├── levels.json               # level_length por nivel
├── requirements.txt
├── README.md
├── dqn/
│   ├── train.py              # Entrenamiento DQN (SB3)
│   └── README.md             # Hiperparámetros y memoria del buffer
├── ppo/                      # (vacío: P1, día 2)
├── app/                      # (vacío: P1, día 3, interfaz)
├── scripts/
│   ├── smoke_test_env.py     # Puerta de entrada
│   └── measure_level_length.py
├── notebooks/
│   ├── colab_train.ipynb     # Lanzador de entrenamiento en Colab
│   └── model_development.ipynb  # Esqueleto del entregable final
├── results/                  # CSV/JSON de evaluación
├── models/                   # checkpoints (NO van a Git; ver §6)
├── runs/                     # logs TensorBoard + episodes.csv
└── documentation/            # este archivo
```

---

## 3. Decisiones tomadas

- **Entorno (env-v1):** SIMPLE_MOVEMENT · frame skip 4 · escala de grises · 84×84 · stack de 4 · observación `(4, 84, 84)` uint8 · recompensa cruda del entorno (sin clipping ni escalado).
- **Versiones (fijas):** `numpy<2`, `gym==0.26.2`, `nes-py==8.2.1`, `gym-super-mario-bros==7.4.0`, `gymnasium==0.29.1`, `stable-baselines3==2.3.2`.
  **No instalar `gym-super-mario-bros` 9.x ni `nes-py` 9.x**: usan la API de Gymnasium y rompen `env.py`; además mezclar versiones entre entrenamiento y evaluación pone en riesgo la comparabilidad.
- **Niveles:** entrenamiento 1-1; no vistos 2-1 y 3-1; opcional 1-2.
- **Flujo híbrido:** se **entrena en Colab** (checkpoints en Drive) y se **evalúa y visualiza en local** (CPU, solo inferencia).
- **Nombres:** corridas `dqn_s<semilla>`; modelos en `models/dqn/<run>/dqn_s<semilla>_<pasos>_steps.zip` y `dqn_s<semilla>_final.zip`.
- **`noop_max` en entrenamiento = 30** (por defecto en `dqn/train.py`). *No está en el plan: confirmar en equipo.*
- **Replay buffer:** en Colab se usa `--optimize-memory-usage` (estimado 2,8 GB en vez de 5,6 GB para 100 000 transiciones).
- **Protocolo de evaluación (en `evaluate.py`):** 1 episodio por semilla; no-ops 0–30 sembrados con la semilla de evaluación; DQN epsilon-greedy fijo 0,05 aplicado en el script (ignora el `exploration_rate` del modelo); PPO muestrea la política; `--mode deterministic` como resultado secundario.

---

## 4. Qué hace cada componente

### `env.py`
`MarioEnv` es un `gymnasium.Env` propio que envuelve el entorno de nes-py (API antigua) usando `.unwrapped`, para que SB3 2.x lo use sin shims. Parámetros: `env_id`, `noop_max`, `max_episode_steps`, `render_mode`. En `info` devuelve `noop_used` al reiniciar.

### `scripts/smoke_test_env.py`
Corre acciones aleatorias en 1-1, 2-1, 3-1 y 1-2; valida forma, dtype y rango de la observación y la terminación de episodios; mide FPS y comprueba que los no-ops son reproducibles con la misma semilla. Termina en `SMOKE TEST OK`.

### `dqn/train.py`
Hiperparámetros de la tabla 4 del plan en un `DQNConfig`, todos modificables por línea de comandos. Opciones relevantes:
`--dry-run` (imprime configuración y memoria estimada, no entrena), `--models-root` / `--logs-root` (separar dónde van modelos y logs), `--resume` (reanudar desde un checkpoint), `--checkpoint-freq`, `--optimize-memory-usage`, `--save-replay-buffer`.
Cada corrida guarda `config.json` (commit, versiones, configuración) y `episodes.csv` (`max_x`, `flag_get`, recompensa, longitud por episodio); en TensorBoard registra `mario/max_x`, `mario/flag_get` y `system/rss_gb`.
Si se reanuda sin replay buffer guardado, ajusta `learning_starts` para volver a llenarlo.

### `evaluate.py`
Ver ejemplos en §5. Guarda `<out>/<tag>.csv` y `.json`. Campos: `algo, seed, level, episode, completed, max_x, steps, reward, elapsed_seconds, prediction_mode` más `eval_seed, checkpoint_steps, noop_max, noop_used, level_length, progress, death_x, end_x, truncated, git_commit`. `run_episode` es reutilizable desde la interfaz. `--record` guarda un mp4 por episodio en `videos/`.

### `scripts/measure_level_length.py`
Se juega el nivel a mano con ventana (necesita WSLg); al tomar la bandera guarda el `x_pos` en `levels.json`.

### `notebooks/colab_train.ipynb`
Lanzador: verifica GPU, monta Drive, clona el repo, instala dependencias (y quita TensorFlow/JAX de Colab), corre el smoke test, hace `--dry-run`, sincroniza logs a Drive cada 5 min, entrena y permite reanudar.

---

## 5. Comandos ejecutados durante el día 1 (referencia)

### 5.1 Entorno local (WSL2 + Ubuntu)
```bash
# PowerShell (administrador)
wsl --install -d Ubuntu

# Ubuntu
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential git
cd "/mnt/c/Users/<usuario>/.../mario-bros-reinforcement-learning"
python3 -m venv ~/venvs/mario          # el venv va FUERA de /mnt/c (más rápido)
source ~/venvs/mario/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Para la ventana de measure_level_length.py
sudo apt install -y libglu1-mesa libgl1 libgl1-mesa-dri mesa-utils
```

### 5.2 Verificaciones
```bash
python scripts/smoke_test_env.py
python -c "from stable_baselines3 import DQN; m = DQN.load('models/dqn/dqn_s1/dqn_s1_final.zip', device='cpu'); print(m.num_timesteps)"   # imprime 50000
python -m dqn.train --seed 1 --total-steps 50000 --dry-run
```

### 5.3 Evaluación
```bash
python evaluate.py --algo random --seeds 100-104
python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --seeds 100-104 --out results/val
python evaluate.py --algo dqn --model models/dqn/dqn_s1/dqn_s1_final.zip --seeds 100-104 --mode deterministic --record
```

### 5.4 Medición de `level_length`
```bash
python scripts/measure_level_length.py --level SuperMarioBros-1-1-v0
python scripts/measure_level_length.py --level SuperMarioBros-2-1-v0
python scripts/measure_level_length.py --level SuperMarioBros-3-1-v0
python scripts/measure_level_length.py --level SuperMarioBros-1-2-v0
```
Valores medidos: 1-1 = 3161, 2-1 = 3193, 3-1 = 3193, 1-2 = 3161 (este último, dudoso; ver §7).

### 5.5 Colab (`notebooks/colab_train.ipynb`)
Orden de celdas: GPU → montar Drive → clonar el repo → instalar dependencias y quitar TF/JAX → smoke test → configuración → `--dry-run` → sincronizar logs → entrenar → (reanudar) → sincronizar y listar Drive.
Comando de entrenamiento que lanza el notebook:
```bash
python -m dqn.train --seed 1 --total-steps 50000 --run-name dqn_s1 \
  --models-root /content/drive/MyDrive/mario/models --logs-root /content/runs \
  --optimize-memory-usage
```

---

## 6. Resultados del smoke test (referencia, no son resultados finales)

| Medición | Valor |
|---|---|
| FPS del entorno solo, Colab (acciones aleatorias) | 92–95 pasos de agente/s |
| FPS del entorno solo, WSL2 local (acciones aleatorias) | 177–196 pasos de agente/s |
| FPS durante el entrenamiento DQN en Colab | ~82–94 (≈82 al final, con entrenamiento activo) |
| Duración de 50 000 pasos de DQN en Colab | ~10 min |
| RAM del proceso en Colab (buffer 100k con `--optimize-memory-usage`) | 1,18 GB a 10k · 1,50 GB a 20k · 3,10 GB a 50k pasos |
| Estimación del buffer | 2,82 GB (con optimize) · 5,64 GB (sin) |

Evaluación en 1-1 con semillas 100–104 (5 episodios, `level_length` aún sin usar para `progress`):

| Agente | `max_x` por semilla | Completado |
|---|---|---|
| Aleatorio | 657, 299, 723, 793, 686 | 0/5 |
| DQN 50k, epsilon-greedy 0,05 | 899, 691, 312, 313, 312 | 0/5 |
| DQN 50k, determinista | 312 ×5 (27 pasos, recompensa ≈249) | 0/5 |

Lectura: con 50 000 pasos el DQN **no supera al aleatorio**. La política determinista repite la misma trayectoria y muere en `max_x` ≈ 312 (parece el primer enemigo; *interpretación sin confirmar con video*). Es lo esperable con tan pocos pasos y no se debe sacar ninguna conclusión; el plan prevé revisar el pipeline el día 4 si el progreso no supera claramente al aleatorio.

---

## 7. Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| `env.reset()` fallaba con gym 0.26 | Los wrappers de `gym.make` asumen la API nueva; nes-py usa la antigua | `.unwrapped` en `env.py` |
| Avisos "Gym has been unmaintained" | `gym 0.26.2` es dependencia de `gym-super-mario-bros 7.4.0` | Inofensivos; no actualizar |
| Aviso "environment out of date, upgrade to v3" | Existen variantes nuevas del entorno | No cambiar: `v0` es la ROM estándar; `v3` usa gráficos simplificados |
| Colab: errores de dependencias con NumPy 2 | Colab trae paquetes que exigen NumPy 2 | Ignorar los avisos de pip; fijamos `numpy<2` a propósito |
| Colab: `import stable_baselines3` fallaba (TensorFlow/JAX) | TensorBoard intenta importar TF, y TF/JAX necesitan NumPy 2 | `pip uninstall -y tensorflow tensorflow-cpu jax jaxlib` (ya está en la celda 4 del notebook) |
| Celda de reanudar: `max() iterable argument is empty` | Con 50k pasos y checkpoint cada 100k no existe ningún checkpoint intermedio | Para pruebas cortas usar `--checkpoint-freq 10000` y otro `RUN_NAME` |
| Windows: `stable_baselines3` no encontrado; no existía `.venv`; sin Python 3.11 | Se instaló en el Python global 3.13 y nes-py/NumPy 1.26 no compilan sin herramientas de C++ | Usar WSL2 con Ubuntu y un venv en `~/venvs/mario` |
| `ImportError: Library "GLU" not found` | Faltan librerías gráficas en Ubuntu/WSL | `sudo apt install -y libglu1-mesa libgl1 libgl1-mesa-dri mesa-utils` |
| `measure_level_length.py` no detectaba la bandera | `play_human` de nes-py no pasa `info` a su callback | El script intercepta `env.step` para leer `info` |
| `python: command not found` | Entorno virtual sin activar | `source ~/venvs/mario/bin/activate` |

---

## 8. Qué debe ejecutar cada integrante para continuar

### 8.1 Preparación (todos, una sola vez)

**Local (Linux o WSL2 con Ubuntu; Windows nativo sin verificar):**
```bash
git clone https://github.com/David-AGL/mario-bros-reinforcement-learning.git
cd mario-bros-reinforcement-learning
python3 -m venv ~/venvs/mario
source ~/venvs/mario/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/smoke_test_env.py            # debe terminar en: SMOKE TEST OK
python -m dqn.train --seed 1 --total-steps 50000 --dry-run
```
Cada vez que abran una terminal nueva: `source ~/venvs/mario/bin/activate` y `cd` al repo.

**Drive compartido (para modelos y logs):**
1. Quien entrena crea `MyDrive/mario/` y la comparte con el equipo como editores.
2. Cada compañero hace clic derecho en esa carpeta → *Organizar* → *Agregar acceso directo* a su Mi unidad, para que `/content/drive/MyDrive/mario` funcione igual en el notebook de todos.
3. Usar un `RUN_NAME` distinto por persona y semilla (p. ej. `dqn_s1`, `ppo_s1`) para no escribir en la misma carpeta.

**Colab:** abrir `notebooks/colab_train.ipynb` (Archivo → Abrir cuaderno → GitHub), activar GPU y ejecutar en orden hasta el smoke test. Anotar los FPS.

### 8.2 Flujo diario con Git
```bash
git pull
# trabajar...
git add <archivos>
git commit -m "mensaje"
git push
```
Antes de subir un notebook ejecutado: *Edit → Clear all outputs*. **Pendiente acordar ramas** (una por persona, o `main` con commits pequeños).

### 8.3 Por rol (día 2 del plan)

**P1 — Entorno, PPO, interfaz**
- Cerrar `env.py` y etiquetar `env-v1` **antes del mediodía** (`git tag env-v1 && git push --tags`). Después, no cambiar el preprocesamiento sin avisar.
- Implementar `ppo/train.py` con el mismo patrón de rutas y registro que `dqn/train.py` (`--models-root`, `--logs-root`, `config.json`, `episodes.csv`). Smoke test de 50k pasos.
- Medir FPS de PPO con 8 entornos y su consumo de RAM.

**P2 — DQN**
- Repetir un smoke test de 50k con checkpoints frecuentes y **probar la reanudación**:
  ```bash
  python -m dqn.train --seed 1 --total-steps 50000 --run-name dqn_smoke2 \
    --checkpoint-freq 10000 --models-root <MODELS_ROOT> --logs-root <LOGS_ROOT> --optimize-memory-usage
  # interrumpir hacia los ~30k pasos y luego reanudar con la celda 9 del notebook
  ```
- Con el presupuesto de pasos acordado, lanzar DQN semilla 1 por la noche.

**P3 — Evaluación**
- Línea base aleatoria en 1-1 con semillas de validación (no evaluar niveles no vistos hasta la fase final):
  ```bash
  python evaluate.py --algo random --seeds 100-104 --out results/val
  ```
- Preparar el script que evalúe cada checkpoint nuevo en 1-1 con las semillas 100–104 y guarde `results/val_*.csv` (día 3).
- Revisar la política de `progress` (§9) y completar la sección del problema en `model_development.ipynb`.

### 8.4 Decisiones que hay que tomar en equipo (día 2)
1. **Presupuesto de pasos** igual para DQN y PPO (referencia en Colab ≈ 82–94 pasos/s: 1 M pasos ≈ 3 h; medir PPO antes de fijarlo).
2. **Entrenar DQN y PPO a la vez o en secuencia**: dos emuladores compiten por CPU y suman RAM.
3. **Ramas de Git.**
4. **`noop_max` en entrenamiento** (hoy 30).
5. **`.gitignore`**: cambiar el actual (que ignora `models/`, `runs/` y `videos/` enteras) por uno que filtre por tipo de archivo y deje versionar `config.json`, `episodes.csv` y los resultados:
   ```
   models/**/*.zip
   models/**/*.pkl
   runs/**/events.out.tfevents.*
   videos/
   ```
   Los modelos se comparten por Drive y los finales por GitHub Releases o Drive (con enlace en el README).

---

## 9. Pendientes y puntos abiertos

- **Reanudación en Colab:** no probada.
- **`level_length` de 1-2:** el valor medido (3161) coincide exactamente con el de 1-1 y ese nivel tiene tuberías y cambios de área, por lo que `max_x / level_length` puede no ser fiable allí. Opciones: dejarlo en `null` y reportar solo la tasa de completado, o validarlo. Los de 2-1 y 3-1 (ambos 3193) conviene verificarlos repitiendo la medición.
- **`progress` en niveles completados:** `max_x / level_length` puede dar algo mayor que 1 por unos píxeles de diferencia. Acordar si se corta en 1,0 o se fija 1,0 cuando `completed = 1`.
- **Ventana de `measure_level_length.py`:** funciona en WSL2 con WSLg tras instalar las librerías gráficas.
- **Videos de `--record`:** usan el códec `mp4v` (15 fps). Se abren en reproductores de escritorio; para Gradio puede hacer falta otro códec (día 3).
- **PPO, interfaz y script de validación por checkpoint:** aún no existen.
- **Memoria en WSL2:** por defecto WSL2 limita la RAM a la mitad de la del equipo (≈8 GB con 16 GB). Tenerlo en cuenta si se entrena en local.
- **Sin verificar:** la ruta de PPO de `evaluate.py` (no hay modelos PPO aún) y el uso de Windows nativo.

---

## 10. Cierre

El día 1 queda **cerrado en lo esencial**: el entorno es reproducible, el pipeline de entrenamiento y evaluación funciona de extremo a extremo (Colab → Drive → local) y existen las bases de cada rol. Antes de lanzar entrenamientos largos faltan: probar la reanudación, acordar ramas y presupuesto de pasos, y etiquetar `env-v1`.
