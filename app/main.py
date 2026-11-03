"""Aplicación en Streamlit para Super Mario Bros RL.

Fase 1 del Plan: Interfaz de Usuario (app/)
Entregable para playtesting, auditoría de agentes y visualización de progreso.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# gdown: descarga modelos desde Google Drive (opcional; se instala si no está)
try:
    import gdown
    _GDOWN_OK = True
except ImportError:
    _GDOWN_OK = False

# Rutas del repositorio
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
VAL_DIR = RESULTS_DIR / "val"
RUNS_DIR = ROOT_DIR / "runs" / "dqn"
VIDEOS_DIR = ROOT_DIR / "videos"
MODELS_DIR = ROOT_DIR / "models" / "dqn"
LEVELS_FILE = ROOT_DIR / "levels.json"

# Asegurar existencia de directorio de videos
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

# ── VIDEOS: Diccionario de Google Drive para despliegue en la nube ──
DRIVE_VIDEOS = {
    "ppo_s1_video_seed200.mp4": "1ztMajZe1E1VBl4CmgHzJnBGWHHIq7aNG",
    "mario_record_78pct_h264.mp4": "18qJ333V6K1Sg9fZNnIFr_XlZtl5DDShI",
    "dqn_1-1_epsilon_greedy_0.05_seeds102-103_seed103.mp4": "1qAKF9Y0KWZMbKV_1lGzE2tPps9o3sVes",
    "video_modelobase.mp4": "1ONijfc4zifJadDD0nRzSYi5gf3EPPlrb",
}

def convert_video_to_h264(video_path: Path):
    """Convierte un video a H.264 para garantizar compatibilidad con navegadores web."""
    raw_bytes = video_path.read_bytes()
    if b"avc1" in raw_bytes:
        return

    tmp_target = video_path.with_name(f"h264_{video_path.name}")
    # 1. Intentar ffmpeg del sistema (en Linux/WSL suele estar disponible)
    import subprocess
    try:
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(tmp_target)]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and tmp_target.exists() and tmp_target.stat().st_size > 1000:
            tmp_target.replace(video_path)
            return
    except Exception:
        pass

    # 2. Intentar imageio si está instalado
    try:
        import imageio.v3 as iio
        frames = iio.imread(video_path)
        iio.imwrite(tmp_target, frames, fps=15, codec="libx264")
        if tmp_target.exists() and tmp_target.stat().st_size > 1000:
            tmp_target.replace(video_path)
            return
    except Exception:
        pass

def play_agent_video(filename: str):
    """Reproduce el video local si existe; si estamos en la nube (ej. Streamlit Cloud), usa el link directo o descarga."""
    local_path = VIDEOS_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 1000:
        st.video(str(local_path))
        return

    # Si no existe localmente pero está registrado en Drive
    fid = DRIVE_VIDEOS.get(filename)
    if fid:
        direct_url = f"https://drive.google.com/uc?export=download&id={fid}"
        try:
            st.video(direct_url)
        except Exception:
            embed_url = f"https://drive.google.com/file/d/{fid}/preview"
            st.components.v1.iframe(embed_url, height=360)
    else:
        st.warning(f"Video {filename} no disponible localmente ni en nube.")

# ── MODELOS: rutas locales + IDs de Google Drive ──────────────────
# Para obtener el FILE_ID: Drive → Share → "Anyone with link" →
# la URL es https://drive.google.com/file/d/<FILE_ID>/view
DRIVE_MODELS = {
    "ppo_s1_3m.zip": {
        "local":    ROOT_DIR / "models" / "ppo" / "ppo_s1_3m.zip",
        "drive_id": "1NQjm5l-eujDtjfpltJneTS8D0pyRNOp_",  # PPO 3M — bandera 100%
        "algo":     "ppo",
    },
    "dqn_advanced_78pct.zip": {
        "local":    ROOT_DIR / "models" / "dqn" / "dqn_advanced_78pct.zip",
        "drive_id": "1gtCGWQGvk5ddqZEdmFqgIOq-pOkXlRdw",  # DQN Advanced 450k — 78%
        "algo":     "dqn",
    },
    "dqn_cnn_900k.zip": {
        "local":    ROOT_DIR / "models" / "dqn" / "dqn_v2_explore05_s1" / "dqn_v2_explore05_900k.zip",
        "drive_id": "1S8taiK1pDBSNEDUnQ-YavfWOCY4RTju1",  # DQN 900k steps
        "algo":     "dqn",
    },
}

def ensure_model(model_key: str) -> Path | None:
    """Devuelve la ruta local del .zip, descargándolo de Drive si hace falta (vía requests o gdown)."""
    cfg = DRIVE_MODELS.get(model_key)
    if cfg is None:
        return None
    local: Path = cfg["local"]
    if local.exists():
        return local
    drive_id = cfg.get("drive_id", "").strip()
    if not drive_id:
        return None

    local.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Descargando {model_key} desde Google Drive..."):
        try:
            if _GDOWN_OK:
                url = f"https://drive.google.com/uc?id={drive_id}"
                gdown.download(url, str(local), quiet=False)
            else:
                import requests
                session = requests.Session()
                url = "https://docs.google.com/uc?export=download"
                response = session.get(url, params={"id": drive_id}, stream=True)
                for k, v in response.cookies.items():
                    if k.startswith("download_warning"):
                        response = session.get(url, params={"id": drive_id, "confirm": v}, stream=True)
                        break
                with open(local, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.error(f"Error al descargar {model_key}: {e}")
            return None

    return local if local.exists() else None

# Añadir raíz al sys.path para reutilizar evaluate y env si están disponibles
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configuración de página
st.set_page_config(
    page_title="Super Mario Bros RL | Interfaz de Evaluación",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS retro / limpios
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E52521;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 5px solid #E52521;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stProgress > div > div > div > div {
        background-color: #E52521;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_levels():
    """Carga longitudes de niveles desde levels.json."""
    if LEVELS_FILE.exists():
        try:
            return json.loads(LEVELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "train": {"SuperMarioBros-1-1-v0": {"level_length": 3161}},
        "unseen": {
            "SuperMarioBros-2-1-v0": {"level_length": 3193},
            "SuperMarioBros-3-1-v0": {"level_length": 3193},
        },
    }


def get_level_length(level_id: str, levels_data: dict) -> int:
    for cat in ["train", "unseen", "optional"]:
        if cat in levels_data and level_id in levels_data[cat]:
            return levels_data[cat][level_id].get("level_length", 3161)
    return 3161


def load_val_csvs():
    """Carga resultados de validación disponibles y normaliza columnas."""
    data = {}
    for f in sorted(glob.glob(str(VAL_DIR / "*.csv"))):
        stem = Path(f).stem
        try:
            df = pd.read_csv(f)
            # Homogeneizar 'flag_get' -> 'completed'
            if "completed" not in df.columns:
                if "flag_get" in df.columns:
                    df["completed"] = df["flag_get"].astype(int)
                else:
                    df["completed"] = (df["max_x"] >= 3161).astype(int)
            # Homogeneizar 'eval_seed'
            if "eval_seed" not in df.columns:
                if "seed" in df.columns:
                    df["eval_seed"] = df["seed"]
                else:
                    df["eval_seed"] = df.index + 100
            # Homogeneizar 'progress'
            if "progress" not in df.columns and "max_x" in df.columns:
                df["progress"] = (df["max_x"] / 3161.0).round(4)
            # Homogeneizar 'steps'
            if "steps" not in df.columns:
                if "length" in df.columns:
                    df["steps"] = df["length"]
                elif "timesteps" in df.columns:
                    df["steps"] = df["timesteps"]
                else:
                    df["steps"] = 0
            data[stem] = df
        except Exception:
            pass
    return data


def safe_show_table(df):
    """Renderiza tablas de forma segura sin disparar el bug de pyarrow con NumPy < 2."""
    try:
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.table(df)


levels_data = load_levels()
val_csvs = load_val_csvs()

# =============================================================
# ENCABEZADO
# =============================================================
st.markdown('<div class="main-title">🍄 Super Mario Bros RL — Panel de Control</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interfaz de evaluación para playtesting automático, métricas de progreso y visualización de partidas.</div>',
    unsafe_allow_html=True,
)

# =============================================================
# BARRA LATERAL: SELECTORES
# =============================================================
st.sidebar.header("🕹️ Configuración del Agente")

# 1. Selector de Agente
agent_choices = {
    "🏆 PPO Campeón (3M pasos — Bandera 100%)": {
        "csv_key": "ppo_s1_3m",
        "algo": "ppo",
        "description": "Campeón indiscutible: completa el nivel 1-1 en 20/20 partidas (100% de éxito) alcanzando la bandera (x=3161). Video: ppo_s1_video_seed200.mp4.",
        "model_file": "ppo_s1_3m.zip",
    },
    "🥈 DQN Advanced / Fine-Tuning (450k pasos — Récord 78%)": {
        "csv_key": "dqn_advanced_phase_tuning_450k",
        "algo": "dqn",
        "description": "Fine-tuning avanzado: progreso medio del 47.6% y pico récord de x=2463 px (78% del nivel). Video: mario_record_78pct_h264.mp4.",
        "model_file": "dqn_advanced_78pct.zip",
    },
    "🥉 DQN CNN Base (900k pasos — 34.5% - 48%)": {
        "csv_key": "dqn_v2_explore05_900k",
        "algo": "dqn",
        "description": "Entrenamiento CNN desde cero (900k pasos): media de x=1114 px (35.2%) con picos de 48% (x=1523 px). Video: dqn_1-1_..._seed103.mp4.",
        "model_file": "dqn_cnn_900k.zip",
    },
    "🎲 Agente Aleatorio (Línea Base)": {
        "csv_key": "random_1-1_uniform_random_seeds100-104",
        "algo": "random",
        "description": "Línea base sin entrenamiento (acciones al azar). Video: video_modelobase.mp4.",
        "model_file": None,
    },
}

# Opción opcional para explorar otros checkpoints intermedios
show_all_checkpoints = st.sidebar.checkbox("Mostrar otros checkpoints intermedios", value=False)
if show_all_checkpoints:
    for k in val_csvs.keys():
        label = f"📁 Checkpoint intermedio: {k}"
        if label not in agent_choices:
            agent_choices[label] = {
                "csv_key": k,
                "algo": "dqn",
                "description": f"Checkpoint de validación {k}",
                "model_file": f"{k}.zip",
            }

selected_agent_label = st.sidebar.selectbox("Seleccionar Agente:", list(agent_choices.keys()), index=0)
selected_agent_info = agent_choices[selected_agent_label]

# 2. Nivel Fijo: Nivel 1-1
selected_level_id = "SuperMarioBros-1-1-v0"
selected_level_label = "Nivel 1-1"
level_length = get_level_length(selected_level_id, levels_data)
st.sidebar.markdown(f"**Nivel de Evaluación:** `Super Mario Bros 1-1`")
st.sidebar.markdown(f"**Distancia a la Bandera:** `{level_length} px`")

# Semilla de evaluación
selected_seed = st.sidebar.selectbox("Semilla de Validación:", [100, 101, 102, 103, 104, "Promedio (Todas)"], index=5)

st.sidebar.markdown("---")
st.sidebar.info(f"**Agente activo:**\n{selected_agent_info['description']}")

# =============================================================
# CUERPO PRINCIPAL: TABS
# =============================================================
tab_main, tab_live, tab_compare = st.tabs([
    "📊 Panel de Métricas y Video",
    "⚡ Modo en Vivo (run_episode)",
    "📈 Comparativa de Modelos",
])

# -------------------------------------------------------------
# TAB 1: PANEL DE MÉTRICAS Y VIDEO PREGRABADO
# -------------------------------------------------------------
with tab_main:
    st.subheader(f"Desempeño en {selected_level_label}")

    df_val = val_csvs.get(selected_agent_info["csv_key"])

    if df_val is not None and not df_val.empty:
        # Filtrar por semilla si se eligió una específica y existe la columna
        if selected_seed != "Promedio (Todas)" and "eval_seed" in df_val.columns:
            df_view = df_val[df_val["eval_seed"] == selected_seed]
            if df_view.empty:
                df_view = df_val.head(1)
        else:
            df_view = df_val

        # Métricas calculadas
        mean_max_x = df_view["max_x"].mean()
        peak_max_x = int(df_view["max_x"].max())
        mean_reward = df_view["reward"].mean()
        progress_pct = (mean_max_x / level_length) * 100
        completed_count = int(df_view["completed"].sum()) if "completed" in df_view.columns else int((df_view["max_x"] >= 3161).sum())
        total_episodes = len(df_view)

        # Muerte más común o última posición (si completó el nivel, death_x viene vacío o NaN)
        if "death_x" in df_view.columns:
            valid_deaths = pd.to_numeric(df_view["death_x"], errors="coerce").dropna().tolist()
        else:
            valid_deaths = []

        if valid_deaths:
            typical_death_str = f"{int(sum(valid_deaths) / len(valid_deaths))} px"
            death_delta = "Promedio"
        elif completed_count == total_episodes:
            typical_death_str = "No murió 🎉"
            death_delta = "100% completado"
        else:
            typical_death_str = f"{int(mean_max_x)} px"
            death_delta = "Última posición"

        # -----------------------------------------------------
        # TARJETAS DE MÉTRICAS EXIGIDAS
        # -----------------------------------------------------
        peak_pct = (peak_max_x / level_length) * 100
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(
                label="Récord Pico Máximo (Mejor Partida)",
                value=f"{peak_pct:.1f}%",
                delta=f"x = {peak_max_x} / {level_length} px",
            )
        with m2:
            st.metric(
                label="Progreso Medio General",
                value=f"{progress_pct:.1f}%",
                delta=f"Media: {mean_max_x:.0f} px",
            )
        with m3:
            st.metric(
                label="Recompensa Acumulada",
                value=f"{mean_reward:.1f} pts",
            )
        with m4:
            bandera_str = "Tomada ✅" if completed_count > 0 else "No tomada ❌"
            st.metric(
                label="Bandera (Completed)",
                value=bandera_str,
                delta=f"{completed_count}/{total_episodes} eps",
            )
        with m5:
            st.metric(
                label="Punto de Muerte (death_x)",
                value=typical_death_str,
                delta=death_delta,
            )

        # Barra visual de progreso (muestra tanto el récord pico como la media)
        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            st.markdown(f"**Avance Máximo de Mario hacia la Bandera (Récord: {peak_pct:.1f}% | Media: {progress_pct:.1f}%):**")
            progress_val = min(max(peak_max_x / level_length, 0.0), 1.0)
            st.progress(progress_val)
        with col_b2:
            st.caption(f"🏁 Meta Bandera: {level_length} px")

        # -----------------------------------------------------
        # VISUALIZADOR DE VIDEO PREGRABADO
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("🎥 Visualizador de Partida en Video (.mp4)")

        col_v1, col_v2 = st.columns([2, 1])

        with col_v1:
            # Videos pregrabados: mapeo por agente + todos los mp4 de videos/
            AGENT_VIDEO_MAP = {
                "ppo_s1_3m": "ppo_s1_video_seed200.mp4",
                "dqn_advanced_phase_tuning_450k": "mario_record_78pct_h264.mp4",
                "dqn_v2_explore05_900k": "dqn_1-1_epsilon_greedy_0.05_seeds102-103_seed103.mp4",
                "dqn_1-1_epsilon_greedy_0.05_seeds100-104": "video_modelobase.mp4",
                "random_1-1_uniform_random_seeds100-104": "video_modelobase.mp4",
            }
            preferred_video = AGENT_VIDEO_MAP.get(selected_agent_info["csv_key"])
            # Listar solo videos reproducibles (excluir archivos raw/no h264)
            local_videos = [p for p in sorted(glob.glob(str(VIDEOS_DIR / "*.mp4"))) if not p.endswith("_raw.mp4")]
            found_videos = local_videos

            # Lista de opciones combinando locales y en la nube
            available_options = list(dict.fromkeys(
                [Path(p).name for p in found_videos] + list(DRIVE_VIDEOS.keys())
            ))

            default_idx = 0
            if preferred_video and preferred_video in available_options:
                default_idx = available_options.index(preferred_video)

            selected_vname = st.selectbox(
                "Seleccionar video pregrabado:",
                available_options,
                index=default_idx,
            )
            play_agent_video(selected_vname)

        with col_v2:
            st.markdown("**Cargar video manual (.mp4):**")
            up_file = st.file_uploader("Arrastra aquí un video de Mario:", type=["mp4"])
            if up_file is not None:
                st.success("Video cargado:")
                st.video(up_file)

        # -----------------------------------------------------
        # TABLA DETALLADA POR SEMILLA
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Detalle de Episodios de Validación")
        disp_cols = ["eval_seed", "max_x", "progress", "reward", "completed", "death_x", "steps", "noop_used"]
        present_cols = [c for c in disp_cols if c in df_view.columns]
        df_show = df_view[present_cols].copy()
        if "progress" in df_show.columns:
            df_show["progress"] = (df_show["progress"] * 100).round(2).astype(str) + "%"

        safe_show_table(
            df_show.rename(columns={
                "eval_seed": "Semilla",
                "max_x": "Max X alcanzado",
                "progress": "Progreso %",
                "reward": "Recompensa",
                "completed": "Bandera",
                "death_x": "Muerte en X",
                "steps": "Pasos",
                "noop_used": "No-ops usados",
            })
        )

    else:
        st.warning(f"No hay archivo CSV de validación para `{selected_agent_info['csv_key']}`.")


# -------------------------------------------------------------
# TAB 2: MODO EN VIVO (RUN_EPISODE REUTILIZADO)
# -------------------------------------------------------------
with tab_live:
    st.subheader("⚡ Ejecución de Episodio en Tiempo Real")
    st.markdown(
        """
        Esta sección reutiliza directamente la función `run_episode` de `evaluate.py` para correr 
        la política seleccionada cuadro a cuadro sobre el emulador y registrar un video recién generado.
        """
    )

    live_seed = st.number_input("Semilla para el episodio en vivo:", min_value=0, max_value=9999, value=100)
    record_live = st.checkbox("Grabar partida a archivo .mp4", value=True)

    if st.button("🚀 Ejecutar Episodio en Vivo", type="primary"):
        with st.spinner("Inicializando emulador y ejecutando episodio..."):
            try:
                # Intentar importar dependencias de simulación
                from evaluate import Policy, load_level_length, load_policy, run_episode
                from env import make_env

                # Verificar si se puede cargar la política
                # Intentar obtener modelo: local o descargando de Drive
                model_path = None
                model_key = selected_agent_info.get("model_file")
                if model_key:
                    resolved = ensure_model(model_key)
                    if resolved is not None:
                        model_path = str(resolved)
                    else:
                        st.info(
                            f"📂 Modelo `{model_key}` no encontrado localmente.\n\n"
                            "Para habilitarlo: abre `app/main.py` y añade el `drive_id` en `DRIVE_MODELS`."
                        )

                # Cargar política
                policy, steps = load_policy(
                    algo=selected_agent_info["algo"],
                    model_path=model_path,
                    mode="stochastic",
                    epsilon=0.05,
                    device="cpu",
                )

                # Crear entorno
                env = make_env(env_id=selected_level_id, noop_max=30, render_mode="rgb_array" if record_live else None)
                rec_path = (VIDEOS_DIR / f"live_{selected_level_id}_seed{live_seed}.mp4") if record_live else None

                res = run_episode(env, policy, eval_seed=int(live_seed), max_steps=4000, record_path=rec_path)
                env.close()

                st.success(f"¡Episodio completado! Max X: {res['max_x']} px | Recompensa: {res['reward']}")

                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.metric("Max X", f"{res['max_x']} px")
                with kpi2:
                    st.metric("Recompensa", f"{res['reward']}")
                with kpi3:
                    st.metric("Pasos", f"{res['steps']}")

                if rec_path and rec_path.exists():
                    convert_video_to_h264(rec_path)
                    st.video(str(rec_path))

            except ImportError as ie:
                st.warning(
                    f"⚠️ Las librerías de emulación NES no están instaladas en este entorno de Python ({ie}).\n\n"
                    "💡 **Solución:** Para correr simulación en vivo, ejecuta Streamlit dentro de tu entorno virtual de WSL Ubuntu (`source ~/venvs/mario/bin/activate`). "
                    "Mientras tanto, puedes visualizar los episodios en la pestaña **'Panel de Métricas y Video'** usando los videos pregrabados."
                )
            except Exception as e:
                st.error(f"Error al ejecutar el episodio en vivo: {e}")


# -------------------------------------------------------------
# TAB 3: COMPARATIVA ENTRE AGENTES + CURVA DE APRENDIZAJE
# -------------------------------------------------------------
with tab_compare:
    st.subheader("📈 Comparativa Directa entre Agentes")

    raw_compare_rows = []
    for label, info in [
        ("1. Agente Aleatorio", "random_1-1_uniform_random_seeds100-104"),
        ("2. DQN CNN Base (900k)", "dqn_v2_explore05_900k"),
        ("3. DQN Advanced / Fine-Tuning (450k)", "dqn_advanced_phase_tuning_450k"),
        ("4. PPO Campeón (3M)", "ppo_s1_3m"),
    ]:
        if info in val_csvs:
            df_i = val_csvs[info]
            prog_val = round((df_i["max_x"].mean() / 3161) * 100, 2)
            raw_compare_rows.append({
                "Agente": label,
                "Max X Medio (px)": round(df_i["max_x"].mean(), 1),
                "Max X Pico (px)": int(df_i["max_x"].max()),
                "Progreso Medio (%)": prog_val,
                "Recompensa Media": round(df_i["reward"].mean(), 1),
            })

    if raw_compare_rows:
        # Orden ascendente estricto por progreso medio
        df_cmp = pd.DataFrame(raw_compare_rows).sort_values("Progreso Medio (%)", ascending=True)
        safe_show_table(df_cmp)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Progreso Medio (% de nivel completado):**")
            st.bar_chart(df_cmp.set_index("Agente")["Progreso Medio (%)"])
        with c2:
            st.markdown("**Recompensa Media:**")
            st.bar_chart(df_cmp.set_index("Agente")["Recompensa Media"])

    # ── Evolución del Agente Activo / Checkpoints ──
    st.markdown("---")
    if selected_agent_info["algo"] == "ppo":
        st.subheader("🏆 Desempeño Consistente de PPO (3M Pasos — 20 Episodios)")
        df_ppo = val_csvs.get("ppo_s1_3m")
        if df_ppo is not None and not df_ppo.empty:
            df_plot = df_ppo[["eval_seed", "max_x", "reward", "steps"]].copy()
            df_plot["Progreso (%)"] = 100.0
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Distancia Máxima alcanzada por Semilla (Meta = 3161 px):**")
                st.bar_chart(df_plot.set_index("eval_seed")["max_x"])
            with col_b:
                st.markdown("**Recompensa Acumulada por Episodio:**")
                st.line_chart(df_plot.set_index("eval_seed")["reward"])
            safe_show_table(df_plot)
    else:
        st.subheader("📉 Curva de Entrenamiento — Evolución por Checkpoints DQN")
        import re as _re
        curve_rows = []
        for key, df_i in sorted(val_csvs.items()):
            m = _re.search(r"explore05_(\d+)k", key)
            if m:
                steps_k = int(m.group(1))
                curve_rows.append({
                    "Pasos (k)": steps_k,
                    "Max X Medio": round(df_i["max_x"].mean(), 1),
                    "Progreso Medio (%)": round((df_i["max_x"].mean() / 3161) * 100, 2),
                    "Recompensa Media": round(df_i["reward"].mean(), 1),
                })

        if curve_rows:
            df_curve = pd.DataFrame(curve_rows).sort_values("Pasos (k)").set_index("Pasos (k)")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Max X Medio por checkpoint:**")
                st.line_chart(df_curve["Max X Medio"])
            with col_b:
                st.markdown("**Progreso Medio (%) por checkpoint:**")
                st.line_chart(df_curve["Progreso Medio (%)"])
            st.markdown("**Recompensa Media por checkpoint:**")
            st.line_chart(df_curve["Recompensa Media"])
            safe_show_table(df_curve.reset_index())
