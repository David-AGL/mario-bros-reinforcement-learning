# Documentación del proyecto

Esta carpeta explica el proyecto `mario-bros-reinforcement-learning` archivo por archivo. La estructura replica la del repositorio para que cada página pueda localizarse junto a su archivo de origen.

## Mapa del proyecto

```text
.
├── app/                  # Paquete reservado para la aplicación
├── dqn/                  # Entrenamiento DQN y su configuración
├── models/               # Checkpoints y metadatos generados (ignorado por Git)
├── notebooks/            # Flujo de entrenamiento en Google Colab
├── ppo/                  # Paquete reservado para el baseline PPO
├── results/              # Resultados versionables
├── runs/                 # Logs de TensorBoard y episodios (generado)
├── scripts/              # Pruebas operativas
├── DIA1_resumen_y_guia.md # Registro operativo y decisiones del día 1
├── env.py                # Entorno MarioEnv
├── evaluate.py           # Evaluación reproducible de agentes
├── levels.json           # Niveles de entrenamiento y evaluación
└── README.md             # Introducción y puesta en marcha
```

## Flujo recomendado

1. Instalar las dependencias del proyecto según el README raíz.
2. Ejecutar el [smoke test](./scripts/smoke_test_env.py.md) para comprobar que el entorno responde.
3. Revisar la configuración y el consumo estimado con el modo `--dry-run` de [dqn/train.py](./dqn/train.py.md).
4. Entrenar desde local o mediante el flujo de [notebooks/colab_train.ipynb](./notebooks/colab_train.ipynb.md).
5. Medir las longitudes faltantes con [scripts/measure_level_length.py](./scripts/measure_level_length.py.md).
6. Evaluar modelos con [evaluate.py](./evaluate.py.md) y analizar los CSV desde [model_development.ipynb](./notebooks/model_development.ipynb.md).
7. Consultar los logs de `runs/` con TensorBoard y conservar los checkpoints de `models/`.

## Contrato del entorno

El preprocesamiento de [env.py](./env.py.md) es `env-v1`: siete acciones `SIMPLE_MOVEMENT`, repetición de cada acción durante cuatro frames, imagen en escala de grises redimensionada a `84x84` y cuatro frames apilados. La observación final es `(4, 84, 84)` de tipo `uint8`.

## Convenciones de esta documentación

- Los archivos Python se explican por responsabilidades, bloques de código, funciones, clases y flujo de ejecución.
- Los notebooks se explican por celda o sección y los scripts de resultados por su esquema y significado.
- Los JSON describen su esquema y el significado de sus campos.
- Los artefactos generados se documentan como salidas, no como código fuente.
- `__init__.py` se incluye aunque esté vacío porque define la intención de cada paquete.
- No se crean páginas independientes para `README.md` ni `requirements.txt`, porque son documentación introductoria y configuración de dependencias respectivamente.
