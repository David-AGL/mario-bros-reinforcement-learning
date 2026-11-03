# Videos generados

La carpeta `videos/` contiene archivos MP4 producidos por `evaluate.py --record`, uno por episodio evaluado.

El nombre sigue el patrón:

```text
<tag>_seed<semilla>.mp4
```

Cada video se genera con `render_mode="rgb_array"`, escala los frames al doble con interpolación de vecino más cercano y se escribe a `15 FPS`. El framerate corresponde a un frame observado por cada paso del agente, ya que el entorno agrupa cuatro frames del NES por acción.

Los MP4 son artefactos binarios para inspección visual y normalmente no deben versionarse en Git. Las métricas reproducibles asociadas se encuentran en los CSV/JSON de `results/`.
