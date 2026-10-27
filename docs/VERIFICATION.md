# Registros de verificación

## Estado de la entrega — 26 de octubre de 2026

La aplicación se desplegó en AWS y se verificó por HTTPS con OpenCV 5.0.0 sobre aarch64. La comparación real terminó y su informe se comprobó en S3 privado y cifrado. La suite actual contiene **60 pruebas aprobadas**. Ver `results/aws_deployment_verification.json` y el índice `results/README.md`.

Para el video se capturó otra comparación: **35,9×** de razón de amplitudes y **43,9 s** de tiempo de ejecución. También se conserva una traza del escenario sintético `fault` que pasa de vigilancia a contraste independiente, generación de evidencia y solicitud de aprobación humana. Ver `results/aws_demo_comparison.json` y `results/aws_agent_trace.json`.

El informe técnico en inglés está en `docs/TECHNICAL_REPORT.md`; la narrativa de Devpost está en `docs/DEVPOST.md`. El registro siguiente corresponde a la revisión anterior del notebook, antes del despliegue AWS.

## Revisión anterior del notebook — 20 de octubre de 2026

- Se revisó el notebook recibido `/Users/admin/Downloads/VibroVision_Colab2.ipynb`, incluidos sus resultados de Colab. Las dos mediciones MIT estaban pendientes porque el perfil usaba tiempos 654/690 s y regiones de otro video.
- Se inspeccionaron y decodificaron directamente los cuatro archivos de `/Users/admin/code/hackaton/videos`. No se usaron videos homónimos de Downloads ni un extracto sustituto.
- `tres.mp4` y `cuatro.mp4` son copias byte a byte: SHA-256 `b6f2beb327fcd569ff5d6f34d1b64c0abddc04985c86ccf3efd25b2aa07a5055`. Duración por cuadros/FPS: 141,2411 s; resolución 1280×720. Se inspeccionaron escenas y rótulos para separar fuente, cámara lenta y amplificación.
- Las 14 celdas de código se ejecutaron con los archivos reales, con `VIBROVISION_VIDEO_DIR` apuntando a esa carpeta. Se conservaron `PROBAR_VENTILADOR=True` y `uno.mp4`/`dos.mp4`. Se omitió sólo la instalación en el entorno local ya preparado.
- El puente original se midió de 100 a 110 s, sin alterar los tiempos para la prueba. Ambos estimadores completaron el seguimiento. Se registran sus discrepancias de series y espectro; no se confirma identificación modal física ni daño estructural.
- Se exporta una sola medición MIT para el contenido único y se conservan los dos nombres de entrada. No se tratan las copias como estados sano/dañado.
- 55 pruebas pasaron, con dos advertencias de deprecación de Starlette. Incluyen IC/LK sobre video sintético conocido, duplicados, perfiles de contenido desconocido, fragmentos fuera de rango y conservación del máximo espectral global.
- Se retiró la celda shell de pip duplicada del notebook recibido. El instalador conserva bibliotecas compatibles y permite una reinstalación explícita de recuperación. OpenCV 5 sigue siendo requerido.
- El notebook limpio pesa aproximadamente 139 KB, sin MP4 ni salidas embebidas. Se conserva una copia ejecutada como evidencia local. Cada ejecución exporta sólo sus resultados nuevos y no incluye videos en el ZIP.
- Las celdas de código entregadas coinciden exactamente con las ejecutadas. La compilación del notebook y su formato se validaron.
- No se autenticó Drive ni se ejecutó esta revisión en Colab en la nube. No se desplegó AWS ni se construyó Docker. Los resultados previos de dos masas son históricos y no se atribuyen a los archivos LISS actuales.

Evidencia actual: `results/liss_bridge_validation.json`, `results/liss_bridge.png`, `results/verification_v3.json`.
