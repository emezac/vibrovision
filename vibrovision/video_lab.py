"""Exploratory multiregion video measurements, independent of the fan geometry.

Operational spectral shapes are descriptive. An educational video without a
documented healthy/damaged pair never becomes a validated damage diagnosis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from scipy.signal import csd, find_peaks
from .tracking import make_tracker

MIT_2DOF_PAGE = ('https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/'
                 'resources/demonstration-of-modes-with-the-2-dof-mechanical-model/')
MIT_2DOF_DOWNLOAD = 'https://archive.org/download/MIT2.003SCF11/MIT2_003SCF11_lec24_300k.mp4'
MIT_LISS_PAGE = 'https://web.mit.edu/liss/research-sensing.html'
LISS_SHOWCASE_SHA256 = 'b6f2beb327fcd569ff5d6f34d1b64c0abddc04985c86ccf3efd25b2aa07a5055'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_liss_inputs(folder, names=('tres.mp4', 'cuatro.mp4')):
    """Content-addressed profile for the actual local LISS montage, not 2DOF."""
    inventory, tasks, pending, seen = [], [], [], {}
    for name in names:
        path = Path(folder) / name
        if not path.is_file():
            raise FileNotFoundError(f'No existe el video: {path}')
        digest = file_sha256(path)
        info = dict(metadata(path), sha256=digest, duplicate_of=seen.get(digest))
        inventory.append(info)
        if digest in seen:
            continue  # Same experiment under another filename; retain both names in inventory.
        seen[digest] = name
        if digest != LISS_SHOWCASE_SHA256:
            pending.append({'file': name, 'status': 'unknown_content_profile',
                            'reason': 'El contenido no coincide con el montaje LISS inspeccionado; requiere otro perfil.'})
            continue
        tasks.append({'name': 'Puente: video original, 100–110 s', 'path': path,
            'start_s': 100., 'duration_s': 10.,
            'regions': {'Union_izquierda': [264, 480, 50, 55],
                        'Union_centro': [785, 480, 60, 58], 'Tablero_centro': [540, 523, 55, 38]},
            'reference': [850, 675, 65, 28], 'axis': 'y', 'method': 'ic',
            'fmin': .5, 'fmax': 8., 'timebase_verified': False, 'source_url': MIT_LISS_PAGE})
    return inventory, tasks, pending


def compare_estimators(primary, traces, alternate, alternate_traces):
    """Descriptive consistency check; does not certify structural modes or damage."""
    from scipy.signal import detrend
    if primary['frames_analyzed'] != alternate['frames_analyzed']:
        raise ValueError('Los estimadores deben usar exactamente los mismos cuadros.')
    axis = 0 if primary['axis'] == 'x' else 1
    channels = {}
    for name, data in traces['displacement'].items():
        if not np.isfinite(data[axis]).all() or not np.isfinite(alternate_traces['displacement'][name][axis]).all():
            channels[name] = {'correlation': None, 'difference_rms_px': None}
            continue
        x, y = detrend(data[axis]), detrend(alternate_traces['displacement'][name][axis])
        denom = float(np.std(x) * np.std(y))
        corr = float(np.mean(x * y) / denom) if denom > 1e-20 else None
        channels[name] = {'correlation': corr,
            'difference_rms_px': float(np.sqrt(np.mean((x - y) ** 2)))}
    a, b = primary['spectrum'], alternate['spectrum']
    available = bool(a and b and primary['measurement_supported'] and alternate['measurement_supported'])
    if available:
        fa, fb = a['shapes'][0]['frequency'], b['shapes'][0]['frequency']
        tolerance = 2 * max(a['frequency_bin_width'], b['frequency_bin_width'])
        peaks_agree = bool(abs(fa - fb) <= tolerance)
    else:
        fa = fb = tolerance = None
        peaks_agree = False
    series_agree = bool(channels and all(v['correlation'] is not None and v['correlation'] >= .8
                                         for v in channels.values()))
    consistent = bool(available and peaks_agree and series_agree)
    if not available:
        interpretation = 'Al menos un seguimiento no está soportado; no se confirma un modo estructural.'
    elif consistent:
        interpretation = 'Los estimadores concuerdan descriptivamente; eso no certifica un modo físico.'
    elif peaks_agree:
        interpretation = 'Los picos coinciden, pero las series no concuerdan en todas las regiones; no se confirma un modo estructural.'
    else:
        interpretation = 'Los picos dominantes no se confirman entre estimadores; no interpretar uno como modo estructural validado.'
    return {'available': available, 'methods': [primary['tracking_method'], alternate['tracking_method']],
            'channels': channels, 'dominant_frequency_primary': fa, 'dominant_frequency_alternate': fb,
            'peak_tolerance': tolerance, 'dominant_peaks_agree': peaks_agree,
            'minimum_correlation_heuristic': .8, 'series_agree': series_agree, 'consistent': consistent,
            'scope': 'Comparación descriptiva: dos bins y correlación 0,8 son heurísticos, no umbrales de daño calibrados.',
            'interpretation': interpretation}


def download_mit_excerpt(destination):
    """Cache only 70 s from the official lecture, with provenance alongside it."""
    destination = Path(destination)
    if destination.exists():
        return destination
    if not shutil.which('ffmpeg'):
        raise RuntimeError('Se necesita FFmpeg para extraer el fragmento. Instálalo o guarda el video en Drive manualmente.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.partial.mp4')
    try:
        completed = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-rw_timeout', '20000000',
            '-ss', '640', '-i', MIT_2DOF_DOWNLOAD, '-t', '70', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-y', str(temporary),
        ], capture_output=True, text=True, timeout=240)
        if completed.returncode:
            raise RuntimeError('No se pudo obtener el fragmento del MIT. Puedes guardarlo en Drive manualmente. ' + completed.stderr[-1200:])
        info = metadata(temporary)
        if not 68 <= info['playback_duration_s'] <= 72:
            raise RuntimeError('El fragmento está incompleto; no se utiliza como si tuviera 70 s.')
        temporary.replace(destination)
        destination.with_suffix('.source.json').write_text(json.dumps({
            'institution': 'MIT OpenCourseWare', 'source_page': MIT_2DOF_PAGE,
            'source_download': MIT_2DOF_DOWNLOAD, 'source_start_s': 640,
            'duration_s': 70, 'processing': 'H.264 CRF18, without audio, no temporal resampling',
            'physical_acquisition_fps_verified': False,
            'attribution': 'MIT OpenCourseWare, J. Kim Vandiver, 2.003SC Engineering Dynamics, Fall 2011',
            'license_page': 'https://ocw.mit.edu/pages/privacy-and-terms-of-use/',
        }, indent=2))
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir el video: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('El contenedor no informa FPS válidos; convierte el archivo antes de medir.')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {'file': Path(path).name, 'container_fps': fps, 'frames': count,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'playback_duration_s': count / fps}
    finally:
        cap.release()


def frame_at(path, time_s=0):
    """Frame in original pixel coordinates; times refer to container playback."""
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError('El tiempo debe ser finito y no negativo.')
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(float(time_s) * fps))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'No hay cuadro en {time_s:g} s. Revisa el inicio del fragmento.')
        return frame
    finally:
        cap.release()


def _boxes(regions, shape):
    if not isinstance(regions, dict) or len(regions) < 2:
        raise ValueError('Marca al menos dos piezas distintas con sus rectángulos [x,y,ancho,alto].')
    h, w = shape[:2]
    checked = {}
    for name, raw in regions.items():
        vals = np.asarray(raw, float)
        if not isinstance(name, str) or not name.strip() or vals.shape != (4,) or not np.isfinite(vals).all():
            raise ValueError('Cada región necesita un nombre y cuatro coordenadas finitas.')
        if not np.equal(vals, np.round(vals)).all():
            raise ValueError('Las coordenadas deben ser enteras, en píxeles del cuadro mostrado.')
        x, y, bw, bh = map(int, vals)
        if min(x, y) < 0 or min(bw, bh) < 12 or x + bw > w or y + bh > h:
            raise ValueError(f'Región fuera de la imagen o menor de 12×12: {name}')
        checked[name] = (x, y, bw, bh)
    if len(set(checked.values())) != len(checked):
        raise ValueError('Dos regiones tienen el mismo rectángulo; selecciona piezas distintas.')
    return checked


def spectral_summary(signals, fs, names, fmin=.1, fmax=None):
    """Welch spectral matrix and dominant operational shapes, without fault labels."""
    x = np.asarray(signals, float)
    if x.ndim != 2 or len(x) != len(names) or x.shape[1] < 64 or not np.isfinite(x).all():
        raise ValueError('Se requieren al menos 64 muestras finitas por región.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('La frecuencia de muestreo debe ser positiva.')
    upper = .45 * fs if fmax is None else min(float(fmax), .45 * fs)
    if not np.isfinite(fmin) or not np.isfinite(upper) or not 0 < fmin < upper:
        raise ValueError('La banda debe ser positiva y quedar por debajo de Nyquist.')
    # At least three overlapping Welch segments, even in short excerpts.
    nper = min(max(32, round(8 * fs)), x.shape[1] // 2)
    spectra = None
    for i in range(len(x)):
        for j in range(i, len(x)):
            f, p = csd(x[i], x[j], fs=fs, nperseg=nper, noverlap=nper // 2, detrend='linear')
            if spectra is None:
                spectra = np.zeros((len(f), len(x), len(x)), complex)
            spectra[:, i, j], spectra[:, j, i] = p, p.conj()
    selected = (f >= fmin) & (f <= upper)
    if selected.sum() < 3:
        raise ValueError('Fragmento demasiado corto para la banda elegida; amplía duración o banda.')
    f, spectra = f[selected], spectra[selected]
    values, vectors = np.linalg.eigh(spectra)
    values = np.maximum(values, 0)
    leading = values[:, -1]
    if leading.max() <= 1e-20:
        raise ValueError('No hay movimiento espectral resoluble en las regiones seleccionadas.')
    peaks, _ = find_peaks(10 * np.log10(np.maximum(leading, 1e-30)), prominence=3)
    candidates = list(peaks)
    # Band selection can put the dominant peak on a boundary; scipy's peak
    # finder excludes endpoints. Keep them explicitly, labeled as band edges.
    if leading[0] > leading[1]:
        candidates.append(0)
    if leading[-1] > leading[-2]:
        candidates.append(len(leading) - 1)
    # A broad global maximum can fail a prominence filter near a band edge.
    # Never call a smaller, sharper peak the dominant spectral maximum.
    global_maximum = int(np.argmax(leading))
    candidates.append(global_maximum)
    peaks = sorted(set(candidates), key=lambda p: (-leading[p], p))[:3]
    if not peaks:
        peaks = [int(np.argmax(leading))]
    shapes = []
    for p in peaks:
        v = vectors[p, :, -1]
        anchor = int(np.argmax(np.abs(v)))
        v = v * np.exp(-1j * np.angle(v[anchor])) / max(np.abs(v).max(), 1e-30)
        power = np.diag(spectra[p]).real
        denom = np.outer(power, power)
        coherence = np.clip(np.abs(spectra[p]) ** 2 / np.maximum(denom, 1e-30), 0, 1)
        shapes.append({'frequency': float(f[p]), 'at_band_edge': bool(p in {0, len(f) - 1}),
                       'is_global_spectral_maximum': bool(p == global_maximum),
                       'relative_amplitude': np.abs(v).tolist(),
                       'phase_deg': np.rad2deg(np.angle(v)).tolist(),
                       'coherence': coherence.tolist(),
                       'leading_power_fraction': float(leading[p] / max(values[p].sum(), 1e-30))})
    return {'frequency': f.tolist(), 'spectral_eigenvalues': values.tolist(),
            'roi_psd': np.diagonal(spectra, axis1=1, axis2=2).real.T.tolist(),
            'frequency_bin_width': float(fs / nper), 'shapes': shapes,
            'scope': 'Formas operacionales descriptivas; un pico también puede proceder de la excitación.'}


def analyze_excerpt(path, regions, *, reference=None, start_s=0, duration_s=30,
                    axis='x', acquisition_fps=None, timebase_verified=False,
                    fmin=.1, fmax=None, source_url=None, method='lk'):
    """Track sparse features incrementally without retaining a full video in RAM.

    Durations/seeks use container seconds; acquisition_fps changes only the
    sampling interval, never which frames are selected. No temporal decimation.
    """
    if axis not in {'x', 'y'} or not np.isfinite(start_s) or start_s < 0:
        raise ValueError('Indica eje x/y y un inicio no negativo.')
    if not np.isfinite(duration_s) or not 0 < duration_s <= 120:
        raise ValueError('Usa un fragmento de hasta 120 segundos.')
    if method not in {'lk', 'ic'}:
        raise ValueError('El estimador debe ser lk o ic.')
    info = metadata(path)
    if start_s + duration_s > info['playback_duration_s'] + 1 / info['container_fps']:
        raise ValueError('El fragmento solicitado excede la duración del archivo.')
    fs = info['container_fps'] if acquisition_fps is None else float(acquisition_fps)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('FPS de adquisición inválidos.')
    first = frame_at(path, start_s)
    boxes = _boxes(regions, first.shape)
    if '__reference__' in boxes:
        raise ValueError('El nombre __reference__ está reservado para la referencia estática.')
    names = list(boxes)
    if reference is not None:
        ref = _boxes({'reference': reference, 'dummy': next(iter(boxes.values()))}, first.shape)['reference']
        boxes['__reference__'] = ref
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    origins, points, good = {}, {}, {}
    for name, (x, y, w, h) in boxes.items():
        mask = np.zeros(gray.shape, np.uint8)
        mask[y:y+h, x:x+w] = 255
        p = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=.02, minDistance=3, mask=mask)
        if p is None or len(p) < 4:
            raise ValueError(f'Poca textura en {name}. Incluye una marca, esquina o borde con textura.')
        origins[name], points[name] = p.copy(), p.copy()
        good[name] = np.ones(len(p), bool)
    displacement = {name: [[0., 0.]] for name in boxes}
    quality = {name: [1.] for name in boxes}
    fixed_trackers = {name: make_tracker('ic', gray, box) for name, box in boxes.items()} if method == 'ic' else {}
    cut_times, previous = [], gray
    cap = cv2.VideoCapture(str(path))
    start_frame = round(start_s * info['container_fps'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)
    limit = min(round(duration_s * info['container_fps']), 20000)
    try:
        for k in range(1, limit):
            ok, frame = cap.read()
            if not ok:
                break
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(cv2.absdiff(previous, current)) > 45:
                cut_times.append(float(k / info['container_fps']))
            if method == 'ic':
                for name, tracker in fixed_trackers.items():
                    dx, dy, score = tracker.track(current)
                    displacement[name].append([dx, dy])
                    quality[name].append(score)
                previous = current
                continue
            for name in boxes:
                valid = np.flatnonzero(good[name])
                if len(valid) >= 4:
                    p = points[name][valid]
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(previous, current, p, None,
                                                         winSize=(21, 21), maxLevel=3)
                    if nxt is None:
                        good[name][:] = False
                    else:
                        back, stb, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None,
                                                              winSize=(21, 21), maxLevel=3)
                        if back is None:
                            good[name][:] = False
                        else:
                            keep = (st.ravel() == 1) & (stb.ravel() == 1)
                            keep &= np.linalg.norm((back - p).reshape(-1, 2), axis=1) < 1.
                            keep &= np.isfinite(nxt.reshape(-1, 2)).all(axis=1)
                            keep &= (nxt[:, 0, 0] >= 0) & (nxt[:, 0, 0] < current.shape[1])
                            keep &= (nxt[:, 0, 1] >= 0) & (nxt[:, 0, 1] < current.shape[0])
                            good[name][valid[~keep]] = False
                            points[name][valid[keep]] = nxt[keep]
                active = np.flatnonzero(good[name])
                shift = (np.median((points[name][active] - origins[name][active]).reshape(-1, 2), axis=0)
                         if len(active) >= 4 else np.array([np.nan, np.nan]))
                displacement[name].append(shift.tolist())
                quality[name].append(float(good[name].mean()) if len(active) >= 4 else 0.)
            previous = current
    finally:
        cap.release()
    disp = {name: np.asarray(rows).T for name, rows in displacement.items()}
    q = {name: np.asarray(rows) for name, rows in quality.items()}
    measured_n = disp[names[0]].shape[1]
    support = all(np.isfinite(d).all() for d in disp.values()) and all(np.min(v) >= .5 for v in q.values()) and not cut_times
    series = {name: disp[name].copy() for name in names}
    if reference is not None:
        series = {name: d - disp['__reference__'] for name, d in series.items()}
    axis_i = 0 if axis == 'x' else 1
    spectrum, rms = None, None
    if support:
        data = np.stack([series[name][axis_i] for name in names])
        spectrum = spectral_summary(data, fs, names, fmin, fmax)
        rms = dict(zip(names, np.std(data, axis=1).tolist()))
    report = {'file': info, 'source_url': source_url, 'sha256': file_sha256(path),
              'start_container_s': float(start_s), 'requested_container_duration_s': float(duration_s),
              'frames_analyzed': measured_n, 'axis': axis, 'regions': boxes, 'tracking_method': method,
              'sampling_fps': fs, 'timebase_verified': bool(timebase_verified),
              'frequency_unit': 'Hz' if timebase_verified else 'ciclos/s de reproducción (no Hz físicos verificados)',
              'quality': {name: {'min_score': float(v.min()), 'median_score': float(np.median(v)),
                                'score_kind': 'surviving_feature_fraction' if method == 'lk' else 'photometric_residual_score'}
                          for name, v in q.items()}, 'suspected_cut_times_s': cut_times,
              'measurement_supported': bool(support), 'rms_px': rms, 'spectrum': spectrum,
              'support_scope': 'Control de seguimiento y de cortes detectados; no certificación física ni de incertidumbre.',
              'camera_translation_compensated': reference is not None,
              'reference_rms_px': float(np.std(disp['__reference__'][axis_i])) if reference is not None else None,
              'structural_verdict': {'decision': 'abstain',
                  'reason': 'Sin estados sano/dañado documentados ni validación de excitación/modelo; no se declara daño ni rigidez.'},
              'limitations': ['LK incremental puede acumular deriva; IC usa una plantilla fija y supone desplazamientos pequeños.',
                             'La referencia sólo compensa traslación común; no rotación o zoom.',
                             'Coherencia y formas operacionales no identifican por sí solas modos estructurales.',
                             'No analizar paneles amplificados ni montajes con cortes como datos físicos.']}
    return report, {'time': np.arange(measured_n) / fs, 'displacement': series,
                    'first_frame': first, 'quality': q,
                    'reference_displacement': disp.get('__reference__')}


def plot_excerpt(report, traces, target):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cv2.cvtColor(traces['first_frame'], cv2.COLOR_BGR2RGB))
    for name, (x, y, w, h) in report['regions'].items():
        axes[0, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='yellow'))
        axes[0, 0].text(x, y, name, color='yellow', backgroundcolor='black')
    axes[0, 0].set_title('Regiones por pieza' + (' y referencia estática' if report['camera_translation_compensated'] else ' (sin compensación de cámara)'))
    a = 0 if report['axis'] == 'x' else 1
    for name, data in traces['displacement'].items():
        axes[0, 1].plot(traces['time'], data[a], label=name)
    axes[0, 1].set(xlabel='Tiempo según base configurada (s)', ylabel='Desplazamiento (px)')
    axes[0, 1].legend()
    spectral = report['spectrum']
    if spectral:
        for i, name in enumerate(traces['displacement']):
            axes[1, 0].semilogy(spectral['frequency'], np.maximum(spectral['roi_psd'][i], 1e-30), label=name)
        axes[1, 0].set(xlabel=report['frequency_unit'], ylabel='PSD (px² por unidad de frecuencia)')
        axes[1, 0].legend()
        names = list(traces['displacement'])
        for shape in spectral['shapes']:
            signed = np.asarray(shape['relative_amplitude']) * np.cos(np.deg2rad(shape['phase_deg']))
            axes[1, 1].plot(names, signed, 'o-', label=f"{shape['frequency']:.3g}")
        axes[1, 1].set(ylabel='Proyección real de la forma relativa', title='Formas operacionales (proyección real)')
        axes[1, 1].legend(title='f (Hz)' if report['timebase_verified'] else 'f (ciclos/s de reproducción)')
    else:
        axes[1, 0].text(.05, .5, 'Medición insuficiente: revisa fragmento y regiones.', transform=axes[1, 0].transAxes)
        axes[1, 1].axis('off')
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
