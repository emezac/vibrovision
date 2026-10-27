#!/usr/bin/env python3
"""Build a light Colab notebook with current sources and videos read from Drive.

All displayed measurements are recomputed. No stored-result fallback is embedded.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    payload = io.BytesIO()
    files = sorted((ROOT / 'vibrovision').glob('*.py'))
    files += [ROOT / 'docs/MATHEMATICS.md']
    manifest = {}
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            data = path.read_bytes()
            name = str(path.relative_to(ROOT))
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    encoded = base64.b64encode(payload.getvalue()).decode()
    archive_hash = hashlib.sha256(payload.getvalue()).hexdigest()
    cells = []

    def md(s):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(s).strip() + '\n'})

    def code(s, hidden=False, when=None):
        source = textwrap.dedent(s).strip() + '\n'
        if when:
            source = f'if {when}:\n' + textwrap.indent(source, '    ') + f"else:\n    print('Sección omitida: {when}=False')\n"
        cells.append({'cell_type': 'code', 'execution_count': None,
                      'metadata': {'jupyter': {'source_hidden': True}} if hidden else {},
                      'outputs': [], 'source': source})

    md('''
    # VibroVision 3 — medir, comprobar y reconocer los límites

    Los videos se leen de **Google Drive → Mi unidad → Colab Notebooks → Vibrovision**.
    Revisa los nombres en la primera celda y elige **Ejecutar todo**; Colab solicitará acceso a Drive.
    No hace falta GitHub ni volver a subir los videos. Este notebook no almacena los MP4.
    El nombre histórico del archivo contiene v2, pero este contenido y sus fuentes corresponden a v3.

    Una cámara puede ver que una máquina se mueve distinto. La matemática ayuda a distinguir ese cambio
    de una diferencia de tamaño aparente, y a reconocer cuándo no hay información para señalar una pieza.
    Aquí comprobamos por separado **álgebra**, **video sintético con verdad conocida** y **videos reales**.
    Esta edición conserva activas las secciones **MIT y ventilador**, con tus cuatro archivos de Drive.
    Puedes desactivar `PROBAR_VENTILADOR` si no quieres ejecutar la comparación de `uno.mp4` y `dos.mp4`.
    ''')
    md('''
    ## Preparar Google Drive

    `CARPETA_VIDEOS` es la ruta dentro de **Mi unidad**, sin incluir «Mi unidad».
    También puedes pegar una ruta absoluta, por ejemplo `/content/drive/Shareddrives/Equipo/videos`.
    Si tus archivos tienen otros nombres, cambia `VIDEO_SIN_MONEDA` y `VIDEO_CON_MONEDA`.
    La comprobación muestra los archivos disponibles si alguno falta; nunca intercambia las condiciones automáticamente.
    No se realiza una búsqueda recursiva en Drive ni se incluyen los videos en las salidas guardadas.
    ''')
    code('''
    #@title Carpeta y nombres de los videos en Google Drive
    CARPETA_VIDEOS = "Colab Notebooks/Vibrovision" #@param {type:"string"}
    PROBAR_MIT = True #@param {type:"boolean"}
    PROBAR_VENTILADOR = True #@param {type:"boolean"}
    VIDEO_SIN_MONEDA = "uno.mp4" #@param {type:"string"}
    VIDEO_CON_MONEDA = "dos.mp4" #@param {type:"string"}

    import os
    from pathlib import Path
    # Esta variable sólo permite verificar el mismo flujo en un entorno local.
    local_drive_root = os.environ.get('VIBROVISION_DRIVE_ROOT')
    local_video_dir = os.environ.get('VIBROVISION_VIDEO_DIR')
    if local_video_dir:
        DRIVE_ROOT = Path(local_video_dir).expanduser().resolve().parent
    elif local_drive_root:
        DRIVE_ROOT = Path(local_drive_root).expanduser().resolve()
    else:
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError('Abre este notebook en Google Colab. Para verificación local, configura VIBROVISION_DRIVE_ROOT.') from exc
        drive.mount('/content/drive')
        DRIVE_ROOT = Path('/content/drive')

    if not CARPETA_VIDEOS.strip():
        raise ValueError('Indica CARPETA_VIDEOS con la ruta de tus videos dentro de Mi unidad.')
    VIDEO_FOLDER = Path(local_video_dir or CARPETA_VIDEOS).expanduser()
    if not local_video_dir and not VIDEO_FOLDER.is_absolute():
        VIDEO_FOLDER = DRIVE_ROOT / 'MyDrive' / VIDEO_FOLDER
    if not VIDEO_FOLDER.is_dir():
        raise FileNotFoundError(f'No existe la carpeta: {VIDEO_FOLDER}. Revisa la cuenta de Drive y CARPETA_VIDEOS.')
    CLIPS, REAL, MIT_RESULTS = [], None, []
    if PROBAR_VENTILADOR:
        if not VIDEO_SIN_MONEDA.strip() or not VIDEO_CON_MONEDA.strip():
            raise ValueError('Indica el nombre de cada video y su condición: sin moneda / con moneda.')
        CLIPS = [VIDEO_FOLDER / VIDEO_SIN_MONEDA, VIDEO_FOLDER / VIDEO_CON_MONEDA]
        missing = [str(path) for path in CLIPS if not path.is_file()]
        if missing:
            available = sorted(p.name for p in VIDEO_FOLDER.iterdir()
                               if p.is_file() and p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'})
            raise FileNotFoundError(f'No se encontraron: {missing}. Videos en esta carpeta: {available[:30]}. Corrige los nombres arriba.')
        if os.path.samefile(*CLIPS):
            raise ValueError('Seleccionaste el mismo archivo para ambas condiciones. Revisa los dos nombres.')
        for label, path in zip(['Sin moneda', 'Con moneda'], CLIPS):
            print(f'{label}: {path} ({path.stat().st_size / 1e6:.2f} MB)')
    print('Carpeta de videos preparada:', VIDEO_FOLDER, '| No se incrustan videos en el notebook.')
    ''')
    code('''
    import os, sys, subprocess
    from importlib.metadata import version, PackageNotFoundError
    REINSTALAR_DEPENDENCIAS = False #@param {type:"boolean"}
    # No reinstalar un entorno compatible en cada ejecución. OpenCV 5 sigue siendo obligatorio.
    if os.environ.get('VIBROVISION_SKIP_INSTALL') != '1':
        required = {'opencv-python-headless': (5, 0), 'numpy': (2, 5), 'scipy': (1, 18), 'matplotlib': (3, 11)}
        installed, compatible = {}, True
        for package, minimum in required.items():
            try:
                installed[package] = version(package)
                pair = tuple(int(v) for v in installed[package].split('.')[:2])
                compatible &= pair[0] == minimum[0] and pair >= minimum
            except (PackageNotFoundError, ValueError):
                compatible = False
        if REINSTALAR_DEPENDENCIAS or not compatible:
            extra = ['--ignore-installed', '--break-system-packages'] if REINSTALAR_DEPENDENCIAS else []
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *extra,
                'opencv-python-headless==5.0.0.93', 'numpy==2.5.3', 'scipy==1.18.1', 'matplotlib==3.11.2'])
            print('Dependencias instaladas. Si ya las habías importado, reinicia la sesión antes de continuar.')
        else:
            print('Se conserva el entorno compatible:', installed)
    ''')
    md('''
    ## Código verificable, videos externos
    La celda plegada contiene solamente el código actual y el documento matemático, verificados por SHA-256.
    Los videos permanecen en Drive; el análisis registra sus hashes para identificar qué archivos se midieron.
    El paquete no contiene videos ni resultados calculados.
    ''')
    code(f'''
import base64, hashlib, io, json, zipfile
from pathlib import Path
PAYLOAD = "{encoded}"
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == "{archive_hash}", "Paquete alterado"
WORK = Path.cwd() / 'vibrovision_colab_v3'
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
    for item in bundle.infolist():
        assert not Path(item.filename).is_absolute() and '..' not in Path(item.filename).parts
    bundle.extractall(WORK)
MANIFEST = {manifest!r}
for name, digest in MANIFEST.items():
    assert hashlib.sha256((WORK / name).read_bytes()).hexdigest() == digest, name
print('Fuentes verificadas por SHA-256; los videos se leerán de Drive. No se cargaron mediciones guardadas.')
del PAYLOAD, raw
''', hidden=True)
    code('''
    sys.path.insert(0, str(WORK))
    import cv2, numpy as np, scipy, matplotlib
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, Image, FileLink
    from vibrovision import __version__
    from vibrovision.validation import algebra_checks, synthetic_video_check, alpha_coverage_check
    from vibrovision.mechanics import compare_operators, modal_sensitivities
    from vibrovision.structure import StructureSpec
    from vibrovision.spectral import alpha_mixture
    from vibrovision.realvideo import compare_clips
    cv2.setNumThreads(1)
    cv2.setRNGSeed(2026)
    import uuid
    RUN_ID = uuid.uuid4().hex[:12]
    RESULTS = WORK / 'results' / RUN_ID
    RESULTS.mkdir(parents=True, exist_ok=False)
    print('VibroVision', __version__, '| OpenCV', cv2.__version__, '| NumPy', np.__version__, '| SciPy', scipy.__version__)
    assert cv2.__version__.startswith('5.'), 'Se requiere OpenCV 5; reinicia el kernel tras instalar si fuera necesario.'
    ''')
    md(r'''
    ## 1. Que una pieza se vea más grande no debe inventar daño

    Cambiamos cien veces la escala de cada región. Comprobamos que el operador mecánico conserva
    sus propiedades. También comprobamos qué movimientos son sensibles a una unión y cómo cambia
    la flexibilidad al perder rigidez. Son propiedades del modelo, no pruebas clínicas o industriales.

    Para quien quiera la fórmula: $D_{ii}=B_{ii}$ y $D_{ij}=-\sqrt{B_{ij}B_{ji}}$ recuperan
    $D=M^{-1/2}KM^{-1/2}$ bajo masas y escalas diagonales positivas y resortes recíprocos.
    ''')
    code('''
    ALGEBRA = algebra_checks()
    display(Markdown(ALGEBRA['explanation']))
    print(json.dumps(ALGEBRA, indent=2, ensure_ascii=False))
    ''')
    md('''
    ## 2. Recuperar una máquina que sí conocemos

    Generamos un video de una estructura simulada, medimos con OpenCV y reconstruimos el operador.
    Podemos comparar con la verdad porque nosotros definimos esa máquina. La incertidumbre se estima
    remuestreando contribuciones de covarianza en bloques temporales, sin barajar cuadros independientes.
    Los radios son empíricos y condicionados al modelo, **no cobertura certificada**.
    ''')
    code('''
    SYNTHETIC = synthetic_video_check(bootstrap_reps=24)
    assert SYNTHETIC['passed'], SYNTHETIC
    estimated = np.asarray(SYNTHETIC['operator'])
    truth = np.asarray(SYNTHETIC['true_operator'])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, matrix, title in zip(axes, [truth, estimated, estimated-truth],
                                  ['Máquina conocida', 'Recuperada desde video', 'Diferencia de estimación']):
        image = ax.imshow(matrix, cmap='coolwarm'); ax.set_title(title); fig.colorbar(image, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(RESULTS / 'operador.png', dpi=140); plt.show()
    print(f"Error relativo en este ensayo: {SYNTHETIC['operator_relative_error']:.2%}")
    print('Incertidumbre:', SYNTHETIC['uncertainty']['method'])
    print('Réplicas válidas:', SYNTHETIC['uncertainty'].get('valid_replicates'))
    print('Disponible:', SYNTHETIC['uncertainty']['available'])
    ''')
    md('''
    ## 3. No siempre podemos poner nombre a una pieza

    Una variación pequeña puede ser indistinguible del error de estimación. Cuando los modos están
    muy próximos, su orientación también es inestable. Comparamos cambios contra radios explícitos;
    cuando no hay un radio disponible, el sistema no declara una localización.
    ''')
    code('''
    D = np.asarray(SYNTHETIC['operator'])
    radius = SYNTHETIC['uncertainty'].get('operator_radius')
    SMALL = compare_operators(D, D * 1.001, radius, radius)
    UNKNOWN = compare_operators(D, D * .8)
    print('Cambio pequeño resuelto:', SMALL['resolved'], '|', SMALL['reason'])
    print('Sin incertidumbre disponible:', UNKNOWN['reason'])
    sp = StructureSpec(); M, C, K, _, _ = sp.matrices({})
    sensitivity = modal_sensitivities(K, M)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, edge in enumerate(sensitivity['edges']):
        ax.plot(np.arange(1, 5), sensitivity['joint'][:, i], 'o-', label=f'Unión {edge[0]}–{edge[1]}')
    ax.set(xlabel='Modo', ylabel='Sensibilidad de frecuencia al cuadrado', title='Cada unión se ve mejor en movimientos diferentes')
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / 'sensibilidades.png', dpi=140); plt.show()
    ''')
    md('''
    ## 4. Ventilador: comparación opcional

    Usa `PROBAR_VENTILADOR=False` en la primera celda si quieres omitir este ensayo.

    Son unos 15 segundos por condición, a aproximadamente 30 fps. La moneda causa una perturbación
    grande. Medimos el aro respecto del fondo. No tenemos varios puntos independientes de toda la
    estructura ni una referencia de fuerza/rigidez. Por eso **no identificamos todos los anclajes**.
    El audio es una corroboración opcional, no verdad calibrada.
    ''')
    code('''
    # Sólo mostramos texto: reproducir con embed=True volvería a guardar el MP4 en el notebook.
    for label, path in zip(['Sin moneda', 'Con moneda'], CLIPS):
        print(f'{label}: {path.name} — se analiza directamente desde Drive.')
    ''', when='PROBAR_VENTILADOR')
    code('''
    cv2.setRNGSeed(2026)
    REAL = compare_clips(CLIPS, out_figure=RESULTS / 'comparacion_real.png')
    display(Image(filename=str(RESULTS / 'comparacion_real.png')))
    rows = ['| Clip | Tono (Hz) | Amplitud tonal (px) | α mediana |', '|---|---:|---:|---:|']
    for clip in REAL['clips']:
        motion, summary = clip['motion'], clip['summary']
        rows.append(f"| {clip['label']} | {motion['tone_hz']:.2f} | {motion['tone_displacement_rms_px']:.3f} | {summary['alpha']['median']:.2f} |")
    display(Markdown('\\n'.join(rows)))
    print(f"Razón de amplitudes tonales: {REAL['tone_amplitude_ratio']:.1f}×")
    print('¿Los rangos de α no se solapan?', REAL['range_separation']['alpha'])
    print('Decisión estructural:', REAL['structural_verdict']['decision'])
    display(Markdown(REAL['explanation']))
    print('Espectro conjunto x/y: distancia descriptiva =', round(REAL['joint_spectral']['shape_distance'], 3))
    print('Son los ejes del mismo aro, no dos piezas. No hay umbral de falla calibrado para esta distancia.')
    ''', when='PROBAR_VENTILADOR')
    md('''
    ## 5. Por qué α puede bajar, subir o no ayudar

    α resume la inclinación del espectro. Añadir energía con un espectro más plano puede bajarla;
    añadir movimiento lento puede subirla. No todas las fallas siguen el mismo patrón. El ejemplo real
    permite reconocer esa limitación. El siguiente gráfico verifica el caso ideal de dos leyes de potencia.
    ''')
    code('''
    f = np.geomspace(2, 55, 200)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for severity in [0, .01, .1, 1]:
        ax.semilogx(f, alpha_mixture(f**-1.6, f**-.3, 1.6, .3, severity), label=f'Aporte añadido {severity}')
    ax.set(xlabel='Frecuencia (Hz)', ylabel='Pendiente local α', title='Si lo añadido es más plano, la pendiente disminuye')
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / 'alpha_condicional.png', dpi=140); plt.show()
    COVERAGE = alpha_coverage_check(200)
    print('Cobertura observada del intervalo en señales ideales:', COVERAGE['empirical_coverage'])
    print(COVERAGE['scope'])
    ''')
    md('''
    ## 6. Videos MIT LISS: analizar el contenido real de tres.mp4 y cuatro.mp4

    Estos archivos contienen el mismo montaje de **141,24 s y 1280×720**, no la clase de dos masas.
    Comprobamos su SHA-256: si son idénticos, se informa la duplicación y el análisis se realiza una sola vez.
    No son dos condiciones experimentales distintas.

    El perfil inspeccionado usa el tramo **100–110 s**, que muestra el puente original. Las regiones evitan
    los rótulos y gráficos superpuestos. Se excluyen la entrevista, la amplificación de color, las diapositivas,
    el tubo reproducido a cámara lenta y el puente amplificado ×400. No mezclamos escenas ni escalas temporales.

    El seguimiento principal IC usa una plantilla fija para movimientos pequeños. Se contrasta con Lucas–Kanade
    sobre exactamente los mismos cuadros. Las frecuencias y formas son descriptivas: codec, cámara, ruido y
    excitación impiden presentar un pico como un modo físico certificado o como evidencia de daño.
    ''')
    code('''
    #@title Videos existentes en Drive (sin descargar ni incrustar archivos)
    ARCHIVOS_MIT = ["tres.mp4", "cuatro.mp4"]
    CONTRASTAR_ESTIMADORES = True #@param {type:"boolean"}
    ''')
    code('''
    from vibrovision.video_lab import (prepare_liss_inputs, analyze_excerpt, frame_at,
                                       plot_excerpt, compare_estimators)
    MIT_INPUTS, MIT_TASKS, MIT_PENDING = [], [], []
    if PROBAR_MIT:
        MIT_INPUTS, MIT_TASKS, MIT_PENDING = prepare_liss_inputs(VIDEO_FOLDER, ARCHIVOS_MIT)
        rows = ['| Archivo | Duración (s) | Resolución | Relación |', '|---|---:|---|---|']
        for entry in MIT_INPUTS:
            relation = ('Copia idéntica de ' + entry['duplicate_of']) if entry['duplicate_of'] else 'Contenido único'
            rows.append(f"| {entry['file']} | {entry['playback_duration_s']:.2f} | {entry['width']}×{entry['height']} | {relation} |")
        display(Markdown('\\n'.join(rows)))
        print('Archivos:', len(MIT_INPUTS), '| Contenidos únicos:', len({e['sha256'] for e in MIT_INPUTS}))
        for task in MIT_TASKS:
            preview = frame_at(task['path'], task['start_s'])
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
            from matplotlib.patches import Rectangle
            for name, (x, y, w, h) in dict(task['regions'], Referencia=task['reference']).items():
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='yellow'))
                ax.text(x, y, name, color='yellow', backgroundcolor='black')
            ax.set(title=task['name'] + ': regiones verificadas en el archivo local', xlabel='x (px)', ylabel='y (px)')
            ax.grid(alpha=.2); plt.show()
        for item in MIT_PENDING:
            print(item['file'], ':', item['reason'])
    ''')
    md('''
    Los tres rectángulos del puente son posiciones de una misma estructura continua, no tres masas aisladas.
    La referencia sólo compensa traslación común y su ruido puede introducir correlación entre canales.
    El material mostrado a ×400 no se utiliza para estimar amplitudes físicas. La indicación «30 fps source»
    se conserva como contexto, sin certificar la adquisición original del montaje descargado.
    ''')
    code('''
    MIT_RESULTS = []
    if PROBAR_MIT:
        for i, task in enumerate(MIT_TASKS):
            options = {k: v for k, v in task.items() if k not in {'name', 'path'}}
            measurement, traces = analyze_excerpt(task['path'], **options)
            measurement['name'] = task['name']
            measurement['input_aliases'] = [item['file'] for item in MIT_INPUTS if item['sha256'] == measurement['sha256']]
            figure = RESULTS / f'mit_{i+1}.png'
            plot_excerpt(measurement, traces, figure)
            display(Image(filename=str(figure)))
            print('Seguimiento aceptado:', measurement['measurement_supported'], '| RMS estimado (px):', measurement['rms_px'])
            if CONTRASTAR_ESTIMADORES:
                alternate_options = dict(options, method='lk')
                alternate, alternate_traces = analyze_excerpt(task['path'], **alternate_options)
                measurement['cross_validation'] = compare_estimators(measurement, traces, alternate, alternate_traces)
                measurement['alternate_estimator'] = alternate
                print(measurement['cross_validation']['interpretation'])
                table = ['| Región | Correlación IC/LK | Diferencia RMS (px) |', '|---|---:|---:|']
                for name, values in measurement['cross_validation']['channels'].items():
                    correlation, difference = values['correlation'], values['difference_rms_px']
                    table.append(f"| {name} | {correlation if correlation is not None else 'No disponible'} | {difference if difference is not None else 'No disponible'} |")
                display(Markdown('\\n'.join(table)))
                np.savez_compressed(RESULTS / f'mit_{i+1}_lk_series.npz', time=alternate_traces['time'],
                                    **{f'disp_{name}': value for name, value in alternate_traces['displacement'].items()})
            if measurement['spectrum']:
                print('Picos descriptivos (no diagnóstico):', [round(s['frequency'], 3) for s in measurement['spectrum']['shapes']], measurement['frequency_unit'])
            print('Diagnóstico:', measurement['structural_verdict']['reason'])
            np.savez_compressed(RESULTS / f'mit_{i+1}_series.npz', time=traces['time'],
                                **{f'disp_{name}': values for name, values in traces['displacement'].items()})
            MIT_RESULTS.append(measurement)
        (RESULTS / 'mit_resultados.json').write_text(json.dumps({'inputs': MIT_INPUTS, 'measurements': MIT_RESULTS,
            'pending': MIT_PENDING}, indent=2, ensure_ascii=False, allow_nan=False))
        if not MIT_RESULTS:
            print('Ningún contenido coincidió con el perfil inspeccionado; no se inventan mediciones.')
    ''')

    md('''
    ## 7. Evidencia exportable y documento matemático

    Se guardan resultados nuevos, versiones y hashes. El documento técnico contiene las demostraciones,
    las condiciones para usar las sumas de filas, la diferencia entre rigidez/amortiguamiento/conectividad
    y los límites del bootstrap y de la identificación pasiva. No se extrapola precisión industrial.
    ''')
    code('''
    import shutil, platform
    REPORT = {'schema_version': '3.0', 'source': 'fresh_notebook_execution', 'run_id': RUN_ID,
              'video_source': {'storage': 'local_directory' if local_video_dir else 'mounted_google_drive',
                               'folder': CARPETA_VIDEOS,
                               'files': sorted(set([p.name for p in CLIPS] + [item['file'] for item in MIT_INPUTS])),
                               'runtime': 'local_verification' if (local_drive_root or local_video_dir) else 'google_colab'},
              'software': {'python': platform.python_version(), 'opencv': cv2.__version__,
                           'numpy': np.__version__, 'scipy': scipy.__version__},
              'algebra': ALGEBRA, 'synthetic_video': SYNTHETIC, 'alpha_intervals': COVERAGE,
              'real_video': REAL, 'mit_videos': MIT_RESULTS, 'mit_pending': MIT_PENDING, 'mit_inputs': MIT_INPUTS,
              'sections_requested': {'fan': PROBAR_VENTILADOR, 'mit': PROBAR_MIT}}
    (RESULTS / 'validacion_colab.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=False, allow_nan=False))
    shutil.copyfile(WORK / 'docs/MATHEMATICS.md', RESULTS / 'MATHEMATICS.md')
    bundle_path = shutil.make_archive(str(WORK / 'evidencia_vibrovision'), 'zip', RESULTS)
    display(FileLink(bundle_path))
    print('En Colab, el paquete también está en el panel Archivos: vibrovision_colab_v3/evidencia_vibrovision.zip')
    # Descarga opcional en Colab:
    # from google.colab import files
    # files.download(bundle_path)
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}, 'colab': {'name': 'VibroVision 3 - comprobación matemática y videos reales'}},
                'cells': cells}
    for i, cell in enumerate(cells):
        cell['id'] = f'vv3-{i:03d}'
    target = ROOT / 'notebooks/vibrometria_visual_alpha_v2.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    print(f'{target.name}: {len(cells)} cells, {target.stat().st_size / 1e3:.0f} KB; no embedded videos; sources verified by SHA-256')


if __name__ == '__main__':
    build()
