#!/usr/bin/env python3
"""Generate the plain-language report from explicitly identified evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'results/math_validation_v3.json'
new = json.loads(path.read_text()) if path.exists() else None
old = json.loads((ROOT / 'results/eval_summary.json').read_text())
deployment_path = ROOT / 'results/aws_deployment_verification.json'
cloud = 'Docker y CloudFormation están preparados; falta registrar un despliegue verificado.'
if deployment_path.exists():
    deployment = json.loads(deployment_path.read_text())
    health = deployment['health']
    cloud = (f"La demo se desplegó y verificó en AWS, región `us-east-2`: {deployment['endpoint']} . "
             f"Ejecuta OpenCV {health['opencv']} sobre {health['arch']} en EC2 Graviton t4g.small. "
             "CloudFront ofrece HTTPS al navegador; el origen EC2 usa HTTP restringido a CloudFront y una cabecera secreta. "
             "La API requiere clave. S3 conserva evidencia privada y cifrada; CloudWatch recibe registros y métricas; "
             "Secrets Manager guarda la clave. No hay suscripción SNS en esta demo. "
             "Las pruebas de autenticación, comparación real y persistencia están en `results/aws_deployment_verification.json`. "
             "La comparación grabada para el video, `results/aws_demo_comparison.json`, tardó 43,9 s y produjo una razón de amplitudes de 35,9×. "
             "Es una ejecución distinta de la validación matemática local. El contenedor usa OpenCV de pip; no se afirma integración COOL ni aceleración medida.")

fresh = 'Aún no ejecutada. Correr `python eval/run_math_validation.py --real`.'
if new:
    a, s = new['algebra'], new['synthetic_video']
    fresh = f'''| Comprobación nueva | Resultado | Qué significa |
|---|---|---|
| 100 cambios de escala por región | Error máximo relativo {a['max_operator_relative_error']:.2g} | Cambiar tamaño aparente no inventa cambio del operador ideal |
| Sensibilidad de modos y pérdida finita | Identidades comprobadas | Podemos explicar qué modos ven cada unión y cómo aumenta la flexibilidad |
| Video sintético → OpenCV → identificación | Error relativo del operador {s.get('operator_relative_error', float('nan')):.2%} | Recuperación en este banco conocido, no precisión industrial |
| Intervalo de α, 200 señales ideales | Cobertura {new['alpha_intervals']['empirical_coverage']:.1%} | Comprobación limitada a leyes de potencia sin tracker ni codec |
'''
    if 'real_video' in new:
        real = new['real_video']
        fresh += f'''
En esta ejecución local de `math_validation_v3.json`, la amplitud del tono cambia aproximadamente **{real['tone_amplitude_ratio']:.1f} veces**. Los rangos de α {'no se solapan' if real['range_separation']['alpha'] else 'se solapan'}. Se conserva la comparación conjunta de los ejes como evidencia descriptiva. La ejecución AWS grabada para la demo se informa por separado.

**Veredicto estructural: abstención.** Sólo medimos un aro y las grabaciones son cortas. No identificamos todos los anclajes, ni desgaste incipiente. El audio, cuando está disponible, es una corroboración; no es verdad calibrada.

![Medición nueva del ventilador](../results/math_validation_v3.png)
'''

report = f'''# VibroVision: qué cambió y por qué importa

## El problema en palabras simples
Una máquina puede cambiar su manera de vibrar antes de dejar de funcionar. Una cámara permite observar varias zonas sin tocarlas. La dificultad no es sólo ver movimiento: hay que distinguir un cambio mecánico de un cambio de cámara o de una medición mala.

VibroVision mide desplazamientos con OpenCV 5, compara la vibración y deja una traza de lo que hizo. Puede cambiar de región, compensar movimiento de cámara, pedir más datos, contrastar un estimador y solicitar revisión humana. No detiene ni controla la máquina.

## Qué conseguimos con la revisión matemática
**Evitar confundir tamaño con daño.** Una pieza más cercana aparece más grande. Bajo el modelo lineal de piezas acopladas, ahora construimos una descripción de la rigidez que conserva sus propiedades aunque cada región tenga una escala distinta.

**No inventar una localización.** Si faltan modos, la reconstrucción no tiene sentido físico o la incertidumbre es demasiado grande, el sistema se abstiene. Puede detectar un cambio sin señalar todavía una pieza concreta.

**Explicar por qué un defecto se ve en unos movimientos y no en otros.** La sensibilidad modal indica qué movimientos cambian cuando pierde rigidez un soporte o una unión. Una fórmula de flexibilidad permite comprobar el efecto de una pérdida finita en un modelo calibrado.

**Escuchar el conjunto, además de una cifra.** La pendiente α resume una parte del espectro, pero no todas las fallas la hacen bajar. Añadimos una comparación del movimiento conjunto entre canales, preservando sus relaciones de fase. Esa distancia todavía es descriptiva, sin probabilidad de falla calibrada.

Las fórmulas y sus condiciones están en [MATHEMATICS.md](MATHEMATICS.md). Los radios de incertidumbre actuales son estimaciones empíricas, no garantías certificadas.

## Qué se ejecutó de nuevo
{fresh}

## Qué significan los videos reales
Los clips incluidos muestran un ventilador sin/con moneda, alrededor de 15 segundos a 30 fps. La moneda es una perturbación grande. Se mide el movimiento del aro respecto del fondo; x/y son dos ejes de esa misma pieza, no dos piezas independientes.

Estos videos ayudan a demostrar una comparación real y un límite útil: podemos mostrar un cambio y, al mismo tiempo, reconocer que no hay datos para reconstruir la máquina completa. Sus ventanas solapadas no son ensayos independientes ni una prueba de generalización.

## Antecedentes de v2, no resultados nuevos de v3
El archivo histórico `results/eval_summary.json` registra RMSE IC {old['estimators']['ic']['rmse_px']:.4f} px en su banco sintético. En las series de prueba, α detectó los 12 fallos con demora mediana {old['matched_fa_5pct']['alpha']['median_delay_s']:.0f} s; la energía alta lo hizo en {old['matched_fa_5pct']['hf_rms']['median_delay_s']:.0f} s. Esos números no se atribuyen automáticamente a la lógica nueva de incertidumbre.

El estudio histórico de vigilancia con h=10 funciona bajo ciertas variaciones de amplitud/carga, pero registra aproximadamente 12,5 alarmas del detector por ROI/día cuando la carga altera la forma del espectro sano. No afirmamos una tasa industrial universal. Alarmas del detector, solicitudes al humano y órdenes aprobadas son cosas diferentes.

## Qué ofrece la demo
La interfaz permite ejecutar escenarios sintéticos, inspeccionar una grabación o comparar ambos clips reales mediante la misma función que usa Colab. Muestra evidencia y solicita aprobación humana. Para análisis estructural real se necesitan regiones manuales que correspondan a piezas distintas; no basta elegir puntos con textura de una sola carcasa.

El notebook es ligero: contiene el código y el documento matemático, y lee los videos desde Google Drive, en `Mi unidad/Colab Notebooks/Vibrovision`. Basta revisar las opciones del ensayo y autorizar el acceso a Drive en Colab. No vuelve a subir los archivos existentes ni guarda los MP4 dentro del notebook. Verifica las fuentes, registra los hashes de los videos y vuelve a calcular los resultados. Exporta un paquete de evidencia con versiones. Un fallo de medición se informa; no se oculta sustituyéndolo por cifras guardadas.

La sección MIT lee `tres.mp4` y `cuatro.mp4` desde Drive, sin nuevas descargas. La inspección de las copias locales mostró que son archivos idénticos de un montaje LISS de 141 segundos. Se analiza una sola vez un tramo original del puente, de 100 a 110 segundos, y se excluyen las escenas amplificadas o ralentizadas. Dos estimadores contrastan el seguimiento de movimientos muy pequeños; si difieren, los picos no se presentan como modos estructurales confirmados. Esta evidencia no acredita daño ni una reconstrucción de rigidez real. Ver [MIT_VIDEOS.md](MIT_VIDEOS.md) para configuración, procedencia y límites.

## Entrega cloud y uso responsable
{cloud}

El servicio procesa las cargas secuencialmente, limita su búfer y borra la carga cruda al terminar. La comparación incluida usa clips acotados en memoria. Las aprobaciones se registran y persisten; no hay control físico automático. Grabar únicamente máquinas con permiso y evitar personas.

## Próximo paso de validación
Repetir con más grabaciones y referencia independiente, variando carga, iluminación, compresión, amortiguamiento, escala y piezas visibles. Medir detección, localización, abstenciones y falsas alarmas por separado. El resultado buscado es una medición que sabe explicar sus límites.
'''
(ROOT / 'docs/REPORT.md').write_text(report)
print('docs/REPORT.md generated from labeled evidence')
