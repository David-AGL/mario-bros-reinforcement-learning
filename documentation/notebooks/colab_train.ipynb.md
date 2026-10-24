# `notebooks/colab_train.ipynb`

Notebook para ejecutar el entrenamiento en Google Colab y evaluar/visualizar localmente.

## Flujo de celdas

1. Explica el flujo híbrido y la necesidad de activar GPU.
2. Comprueba GPU y CUDA con `nvidia-smi` y PyTorch.
3. Monta Google Drive.
4. Clona o actualiza este repositorio desde GitHub.
5. Instala las dependencias compatibles sin reemplazar el PyTorch de Colab.
6. Ejecuta el smoke test y permite comparar FPS.
7. Define `SEED`, `TOTAL_STEPS`, nombre de corrida y rutas de modelos/logs.
8. Desinstala paquetes que pueden generar conflictos y verifica que SB3 importe.
9. Ejecuta `dqn.train --dry-run` para revisar configuración y memoria.
10. Sincroniza logs locales a Drive cada cinco minutos mediante un hilo y `rsync`.
11. Ejecuta el entrenamiento DQN con `--optimize-memory-usage`.
12. Explica cómo buscar el último checkpoint y reanudar con el mismo nombre y presupuesto.
13. Lista checkpoints y sincroniza los logs finales.
14. Indica descargar `models/` y `runs/` a local para TensorBoard y evaluación.

El notebook no implementa una política distinta: orquesta el mismo `dqn.train` y cambia principalmente el almacenamiento entre Drive y el disco efímero de Colab.

