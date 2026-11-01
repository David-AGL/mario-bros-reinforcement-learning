# `ppo/train.py`

Módulo para el entrenamiento de agentes de **Super Mario Bros** utilizando el algoritmo **PPO** (Proximal Policy Optimization) con múltiples entornos paralelos mediante `SubprocVecEnv`.

---

## Fundamento y Ventajas sobre DQN

1. **Eficiencia de Muestreo (Sample Efficiency):** A diferencia de DQN que recopila experiencia secuencialmente con 1 solo entorno (~75 FPS), PPO vectorizado ejecuta 8 entornos simultáneamente en procesos independientes, alcanzando entre **800 y 1,200 FPS**.
2. **Estabilidad On-Policy:** Optimiza directamente la política mediante recorte de ventaja (*clipped surrogate objective*), eliminando la inestabilidad de sobreestimación de valores Q y buffers de repetición obsoletos ante caídas repentinas en fosos.
3. **Objetivo 100% de Completitud:** Permite alcanzar 2,000,000 de pasos en aproximadamente 35 a 45 minutos en Google Colab con aceleración GPU (T4), facilitando la convergencia para capturar la bandera en $x \approx 3,161$ px.

---

## Parámetros CLI

| Argumento | Tipo | Por defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `--total-steps` | `int` | `2_000_000` | Presupuesto total de timesteps de interacción. |
| `--n-envs` | `int` | `8` | Número de procesos en paralelo (`SubprocVecEnv`). |
| `--seed` | `int` | `1` | Semilla del generador pseudoaleatorio. |
| `--lr` | `float` | `2.5e-4` | Tasa de aprendizaje inicial de PPO. |
| `--level` | `str` | `SuperMarioBros-1-1-v0` | Nivel del entorno. |
| `--models-root` | `str` | `./models/ppo` | Carpeta destino para los checkpoints (`.zip`). |
| `--logs-root` | `str` | `./runs/ppo` | Directorio de logs de TensorBoard y métricas de episodios. |
| `--run-name` | `str` | `ppo_s1` | Identificador único de la corrida. |

---

## Uso Típico

### En Google Colab con Google Drive:
```bash
python -m ppo.train \
    --total-steps 2000000 \
    --n-envs 8 \
    --seed 1 \
    --lr 2.5e-4 \
    --models-root /content/drive/MyDrive/mario/models/ppo \
    --logs-root /content/drive/MyDrive/mario/runs/ppo \
    --run-name ppo_mario_100pct
```

### Evaluación posterior con `evaluate.py`:
```bash
python evaluate.py \
    --algo ppo \
    --model "/content/drive/MyDrive/mario/models/ppo/ppo_mario_100pct/ppo_mario_final.zip" \
    --seeds 100-104 \
    --mode deterministic \
    --record \
    --out results/eval_ppo
```
