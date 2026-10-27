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


@st.cache_data
def load_val_csvs():
    """Carga resultados de validación disponibles."""
    data = {}
    for f in sorted(glob.glob(str(VAL_DIR / "*.csv"))):
        stem = Path(f).stem
        try:
            data[stem] = pd.read_csv(f)
        except Exception:
            pass
    return data


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

# 1. Selector de Agente (Los 3 principales exigidos por el plan)
agent_choices = {
    "🌟 DQN Mejorado (900k pasos - explore 0.5) (el mejor modelo)": {
        "csv_key": "dqn_v2_explore05_900k",
        "algo": "dqn",
        "description": "Mejor política aprendida (~48% del nivel en pico). Supera tuberías iniciales y Goombas.",
        "model_file": "dqn_v2_explore05_900k.zip",
    },
    "🐢 DQN Inicial (50k pasos) (línea base DQN)": {
        "csv_key": "dqn_1-1_epsilon_greedy_0.05_seeds100-104",
        "algo": "dqn",
        "description": "Primer intento de entrenamiento sin bonos. Se estanca frente al primer obstáculo.",
        "model_file": "dqn_s1_final.zip",
    },
    "🎲 Agente Aleatorio (línea base sin entrenamiento)": {
        "csv_key": "random_1-1_uniform_random_seeds100-104",
        "algo": "random",
        "description": "Acciones uniformes al azar para fijar el umbral mínimo que debe superar el aprendizaje.",
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

# 2. Selector de Nivel
level_options = {
    "Nivel 1-1 (visto)": "SuperMarioBros-1-1-v0",
    "Nivel 2-1 (no visto)": "SuperMarioBros-2-1-v0",
    "Nivel 3-1 (no visto)": "SuperMarioBros-3-1-v0",
}
selected_level_label = st.sidebar.selectbox("Seleccionar Nivel de Mario:", list(level_options.keys()), index=0)
selected_level_id = level_options[selected_level_label]
level_length = get_level_length(selected_level_id, levels_data)

st.sidebar.markdown(f"**Longitud del nivel hasta la bandera:** `{level_length} px`")

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
        # Filtrar por semilla si se eligió una específica
        if selected_seed != "Promedio (Todas)":
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
        completed_count = int(df_view["completed"].sum())
        total_episodes = len(df_view)

        # Muerte más común o última posición
        death_locations = df_view["death_x"].tolist() if "death_x" in df_view.columns else [mean_max_x]
        typical_death = int(sum(death_locations) / len(death_locations))

        # -----------------------------------------------------
        # TARJETAS DE MÉTRICAS EXIGIDAS
        # -----------------------------------------------------
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric(
                label="Progreso (max_x / level_length)",
                value=f"{progress_pct:.1f}%",
                delta=f"{mean_max_x:.0f} / {level_length} px",
            )
        with m2:
            st.metric(
                label="Coordenada Máxima (max_x)",
                value=f"{peak_max_x} px",
                delta=f"Pico: {(peak_max_x / level_length) * 100:.1f}%",
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
                value=f"{typical_death} px",
                delta="Promedio",
            )

        # Barra visual de progreso
        st.markdown("**Avance de Mario respecto a la meta (Bandera en el asta):**")
        progress_val = min(max(mean_max_x / level_length, 0.0), 1.0)
        st.progress(progress_val)

        # -----------------------------------------------------
        # VISUALIZADOR DE VIDEO PREGRABADO
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("🎥 Visualizador de Partida en Video (.mp4)")

        col_v1, col_v2 = st.columns([2, 1])

        with col_v1:
            # Buscar videos locales en carpeta videos/ o Downloads
            local_videos = sorted(glob.glob(str(VIDEOS_DIR / "*.mp4")))
            downloads_dir = Path.home() / "Downloads"
            wa_videos = sorted(glob.glob(str(downloads_dir / "*WhatsApp Video 2026-10-25*.mp4")))
            found_videos = local_videos + wa_videos

            if found_videos:
                video_map = {Path(p).name: p for p in found_videos}
                selected_vname = st.selectbox("Seleccionar video pregrabado:", list(video_map.keys()))
                video_path_to_play = video_map[selected_vname]
                st.video(video_path_to_play)
            else:
                st.info(
                    "📁 No se detectaron videos `.mp4` en `videos/`. Puedes arrastrar un video en el panel derecho para visualizarlo."
                )

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

        st.dataframe(
            df_show.rename(columns={
                "eval_seed": "Semilla",
                "max_x": "Max X alcanzado",
                "progress": "Progreso %",
                "reward": "Recompensa",
                "completed": "Bandera",
                "death_x": "Muerte en X",
                "steps": "Pasos",
                "noop_used": "No-ops usados",
            }),
            use_container_width=True,
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
                model_path = None
                if selected_agent_info["model_file"]:
                    potential_path = MODELS_DIR / selected_agent_info["model_file"]
                    if potential_path.exists():
                        model_path = str(potential_path)

                # Cargar política
                policy, steps = load_policy(
                    algo=selected_agent_info["algo"],
                    model_path=model_path,
                    mode="stochastic",
                    epsilon=0.05,
                    device="cpu",
                )

                # Crear entorno
                env = make_env(env_id=selected_level_id, noop_max=30)
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
# TAB 3: COMPARATIVA ENTRE LOS 3 AGENTES
# -------------------------------------------------------------
with tab_compare:
    st.subheader("📈 Comparativa Directa entre Agentes")

    compare_rows = []
    for label, info in [
        ("Agente Aleatorio", "random_1-1_uniform_random_seeds100-104"),
        ("DQN Inicial (50k)", "dqn_1-1_epsilon_greedy_0.05_seeds100-104"),
        ("DQN Mejorado (900k)", "dqn_v2_explore05_900k"),
        ("DQN Colapso (1000k)", "dqn_v2_explore05_1000k"),
    ]:
        if info in val_csvs:
            df_i = val_csvs[info]
            compare_rows.append({
                "Agente": label,
                "Max X Medio (px)": round(df_i["max_x"].mean(), 1),
                "Max X Pico (px)": int(df_i["max_x"].max()),
                "Progreso Medio (%)": round((df_i["max_x"].mean() / 3161) * 100, 2),
                "Recompensa Media": round(df_i["reward"].mean(), 1),
            })

    if compare_rows:
        df_cmp = pd.DataFrame(compare_rows)
        st.dataframe(df_cmp, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Progreso Medio (% de nivel completado):**")
            st.bar_chart(df_cmp.set_index("Agente")["Progreso Medio (%)"])
        with c2:
            st.markdown("**Recompensa Media:**")
            st.bar_chart(df_cmp.set_index("Agente")["Recompensa Media"])
