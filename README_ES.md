# VibroVision 3: mirar una máquina y saber qué podemos concluir

Usamos video y OpenCV 5 para medir movimiento. El sistema compara la vibración, comprueba la medición y pide revisión humana cuando encuentra evidencia suficiente. No controla la máquina.

## Qué cambió, sin fórmulas

Una pieza puede verse más grande porque está más cerca de la cámara. Eso no significa que esté más dañada. Ahora construimos una descripción de la rigidez que conserva sus propiedades aunque cada región tenga una escala diferente, bajo el modelo de piezas unidas por resortes.

Separamos tres preguntas: **¿cambió la rigidez?, ¿cambió la forma de vibrar?, ¿podemos señalar una pieza?** A veces el video muestra un cambio, pero no alcanza para localizarlo. El sistema lo dice y pide más información. Comprueba si la reconstrucción tiene sentido físico y estima cuánto puede variar por la medición.

La explicación de resultados está en [el reporte general](docs/REPORT.md). Las demostraciones, hipótesis y límites están en [el documento matemático](docs/MATHEMATICS.md).

## Los videos disponibles

Incluimos las grabaciones de ventilador sin/con moneda: unos 15 segundos a 30 fps. Medimos el movimiento del aro respecto del fondo y comparamos tonos y espectro conjunto de sus ejes. La moneda produce una diferencia; **no demuestra detección temprana de desgaste ni permite identificar todos los soportes**. La pendiente α no separa sus rangos: otros indicadores sí aportan evidencia.

El banco sintético permite comparar con rigideces conocidas. Las cifras antiguas de `results/*summary.json` son antecedentes de v2; no son una evaluación nueva de v3.

## Google Colab

Abrir `notebooks/vibrometria_visual_alpha_v2.ipynb` en Google Colab. Conservamos el nombre para no romper referencias; el contenido es v3. El notebook contiene el código y el documento matemático, **sin videos embebidos**.

1. La carpeta ya está configurada como `Colab Notebooks/Vibrovision`, dentro de **Mi unidad**. Esta edición activa `PROBAR_MIT=True`; la comparación del ventilador queda disponible con `PROBAR_VENTILADOR=True`.
2. Elige **Ejecutar todo** y autoriza el montaje de tu Drive en Colab. Se leen los archivos `tres.mp4` y `cuatro.mp4`, ya guardados en esa carpeta; no se descarga material nuevo.
3. El perfil reconoce por SHA-256 los archivos LISS inspeccionados localmente: duran 141,24 s y contienen un montaje, no la clase de dos masas. `tres.mp4` y `cuatro.mp4` son copias idénticas: se registran ambos nombres y se analiza una sola vez el tramo original del puente de 100 a 110 s. Se comparan IC y Lucas–Kanade sin tratar la amplificación ×400 como datos físicos. Ver [instrucciones de MIT](docs/MIT_VIDEOS.md).
4. El análisis verifica las fuentes y registra los hashes de los videos. Genera gráficas y un ZIP con resultados nuevos, series medidas y documento matemático. Los MP4 tampoco se incrustan al guardar las salidas.

La configuración conserva `PROBAR_VENTILADOR=True`, `VIDEO_SIN_MONEDA="uno.mp4"` y `VIDEO_CON_MONEDA="dos.mp4"` del notebook actualizado por el usuario. No necesita GitHub ni sustituye mediciones por cifras guardadas. El análisis del ventilador puede funcionar sin su corroboración de audio. Para verificación local, `VIBROVISION_VIDEO_DIR=/Users/admin/code/hackaton/videos` evita montar Drive.

Las dependencias compatibles se conservan. Hay una sola celda de instalación; `REINSTALAR_DEPENDENCIAS=True` habilita la recuperación de un entorno roto. No se usan celdas de shell con requisitos `>=` sin comillas.

## Ejecutar y comprobar

Requiere Python 3.12 o superior, según las versiones fijadas de NumPy y SciPy.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python eval/run_math_validation.py --real
python docs/make_report.py
uvicorn service.app:app --port 8000
```

En `http://localhost:8000`, pulsar **Compare the two real fan videos**. Recalcula la comparación con la misma función que Colab. Un video individual sirve para inspección dentro de esa grabación; no equivale a comparar un estado sano con uno dañado.

Regenerar el notebook tras cambios: `python notebooks/build_notebook.py`.

## Desplegar en AWS

La demo ya está publicada en https://d30icy4clt3p0c.cloudfront.net/ y verificada con OpenCV 5.0.0 sobre aarch64 en `us-east-2`. Utiliza una instancia Graviton t4g.small, HTTPS de CloudFront, clave en Secrets Manager, evidencia privada en S3 y registros en CloudWatch. Los comandos siguientes crean un despliegue propio; no actualizan implícitamente la demo existente.

```bash
aws login --profile vibrovision --region us-east-2
python3 infra/deploy_aws.py plan
# Revisar los conjuntos de cambios antes de aprovisionar.
python3 infra/deploy_aws.py execute
python3 infra/deploy_aws.py verify
```

No necesita GitHub ni Docker local. El arranque verifica la huella del código y la salud de OpenCV antes de confirmar que la aplicación está lista. La clave de acceso se guarda localmente en un archivo privado excluido de Git. Ver [la guía de despliegue](DEPLOYMENT_AWS.md) para arquitectura, costos por uso, administración y límites.

Se conserva `infra/deploy.sh` como alternativa anterior basada en un repositorio Git. La plantilla nueva no afirma rendimiento COOL ni sensibilidad industrial. El análisis de videos conserva sus comprobaciones y sus límites matemáticos.

## Código

| Archivo | Función |
|---|---|
| `vibrovision/mechanics.py` | Operador invariante, validez física, cortes, sensibilidad y flexibilidad |
| `vibrovision/structure.py` | Identificación modal, incertidumbre empírica y abstención |
| `vibrovision/spectralmatrix.py` | Movimiento conjunto entre canales |
| `vibrovision/validation.py` | Comprobaciones compartidas con Colab |
| `vibrovision/realvideo.py` | Comparación real |
| `vibrovision/video_lab.py` | Fragmentos genéricos, seguimiento multirregión y formas operacionales para los ejemplos MIT |
| `service/`, `infra/` | Interfaz, aprobación y preparación del despliegue |

Grabar máquinas con permiso. Las cargas crudas se eliminan tras procesarse; se conservan resultados/evidencia y S3 caduca a 30 días. Los clips incluidos son activos explícitos de demostración.
