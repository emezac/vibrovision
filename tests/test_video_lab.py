"""Measurement checks for generic videos: phase, time base and failure handling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest

from vibrovision.video_lab import (analyze_excerpt, compare_estimators, frame_at,
                                  prepare_liss_inputs, spectral_summary)


@pytest.fixture(scope='module')
def moving_video(tmp_path_factory):
    path = tmp_path_factory.mktemp('modal') / 'two_pieces.avi'
    rng = np.random.default_rng(8)
    base = np.uint8(rng.integers(30, 180, (140, 240)))
    patches = [np.uint8(rng.integers(30, 240, (26, 26))) for _ in range(2)]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30., (240, 140))
    assert writer.isOpened()
    for i in range(360):
        image = base.copy()
        camera = .6 * np.sin(2 * np.pi * .7 * i / 30)
        shift = 5 * np.sin(2 * np.pi * 2.5 * i / 30)
        for patch, x, sign in zip(patches, [45, 150], [1, -1]):
            matrix = np.float32([[1, 0, sign * shift], [0, 1, 0]])
            plane = np.zeros_like(image)
            plane[55:81, x:x+26] = patch
            mask = np.zeros_like(image)
            mask[55:81, x:x+26] = 255
            moved = cv2.warpAffine(plane, matrix, (240, 140))
            moved_mask = cv2.warpAffine(mask, matrix, (240, 140))
            image[moved_mask > 0] = moved[moved_mask > 0]
        image = cv2.warpAffine(image, np.float32([[1, 0, camera], [0, 1, 0]]), (240, 140), borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def test_video_recovers_frequency_phase_and_reference(moving_video):
    result, _ = analyze_excerpt(moving_video, {'A': [45, 55, 26, 26], 'B': [150, 55, 26, 26]},
                                reference=[15, 15, 25, 25], duration_s=10, timebase_verified=True,
                                fmin=.2, fmax=7)
    assert result['measurement_supported']
    peak = result['spectrum']['shapes'][0]
    assert abs(peak['frequency'] - 2.5) < .25
    assert abs(abs(peak['phase_deg'][1] - peak['phase_deg'][0]) - 180) < 15
    assert peak['coherence'][0][1] > .9
    assert 2.6 < result['rms_px']['A'] < 4.3
    assert result['structural_verdict']['decision'] == 'abstain'


def test_acquisition_fps_changes_units_without_resampling(moving_video):
    regions = {'A': [45, 55, 26, 26], 'B': [150, 55, 26, 26]}
    normal, _ = analyze_excerpt(moving_video, regions, duration_s=8, fmin=.2, fmax=10)
    faster, _ = analyze_excerpt(moving_video, regions, duration_s=8, acquisition_fps=60,
                                fmin=.2, fmax=10, timebase_verified=True)
    assert normal['frames_analyzed'] == faster['frames_analyzed']
    assert not normal['timebase_verified']
    assert faster['spectrum']['shapes'][0]['frequency'] == pytest.approx(2 * normal['spectrum']['shapes'][0]['frequency'])


def test_different_operational_shapes_with_equal_channel_energy():
    fs = 30
    t = np.arange(1800) / fs
    a = np.sin(2 * np.pi * 2 * t)
    in_phase = spectral_summary(np.stack([a, a]), fs, ['A', 'B'])
    anti_phase = spectral_summary(np.stack([a, -a]), fs, ['A', 'B'])
    np.testing.assert_allclose(in_phase['roi_psd'], anti_phase['roi_psd'])
    assert abs(in_phase['shapes'][0]['phase_deg'][1]) < 1
    assert abs(abs(anti_phase['shapes'][0]['phase_deg'][1]) - 180) < 1


def test_dominant_peak_at_band_boundary_is_not_silently_dropped():
    fs = 30
    t = np.arange(150) / fs
    a = np.sin(2 * np.pi * .8 * t) + .15 * np.sin(2 * np.pi * 2.8 * t)
    result = spectral_summary(np.stack([a, 2 * a]), fs, ['A', 'B'], fmin=.5, fmax=6)
    assert result['shapes'][0]['frequency'] == pytest.approx(.8)
    assert result['shapes'][0]['at_band_edge']


def test_broad_spectral_maximum_is_not_replaced_by_a_smaller_sharp_peak():
    fs = 30
    t = np.arange(300) / fs
    signal = 10 * np.sin(2 * np.pi * .8 * t) + .1 * np.sin(2 * np.pi * 4.4 * t)
    result = spectral_summary(np.stack([signal, .4 * signal]), fs, ['A', 'B'], fmin=.5, fmax=8)
    leading = np.asarray(result['spectral_eigenvalues'])[:, -1]
    expected = result['frequency'][int(np.argmax(leading))]
    assert result['shapes'][0]['frequency'] == expected
    assert result['shapes'][0]['is_global_spectral_maximum']


@pytest.mark.parametrize('regions', [
    {'A': [45, 55, 26, 26]},
    {'A': [45, 55, 26, 26], 'B': [45, 55, 26, 26]},
    {'A': [45, 55, 26, 26], 'B': [220, 120, 26, 26]},
])
def test_invalid_regions_fail_explicitly(moving_video, regions):
    with pytest.raises(ValueError):
        analyze_excerpt(moving_video, regions, duration_s=8)


def test_no_frame_outside_video(moving_video):
    with pytest.raises(ValueError, match='No hay cuadro'):
        frame_at(moving_video, 1000)


def test_scene_cut_and_tracking_loss_withhold_spectrum(moving_video, tmp_path):
    broken = tmp_path / 'cut.avi'
    cap = cv2.VideoCapture(str(moving_video))
    writer = cv2.VideoWriter(str(broken), cv2.VideoWriter_fourcc(*'MJPG'), 30., (240, 140))
    for i in range(180):
        ok, frame = cap.read()
        assert ok
        writer.write(frame if i < 90 else np.full_like(frame, 255))
    writer.release()
    cap.release()
    result, _ = analyze_excerpt(broken, {'A': [45, 55, 26, 26], 'B': [150, 55, 26, 26]}, duration_s=6)
    assert not result['measurement_supported']
    assert result['spectrum'] is None
    assert result['suspected_cut_times_s']


def test_fixed_template_recovers_small_motion_and_matches_frequency(moving_video):
    kwargs = dict(reference=[15, 15, 25, 25], duration_s=10, fmin=.2, fmax=7)
    regions = {'A': [45, 55, 26, 26], 'B': [150, 55, 26, 26]}
    primary, traces = analyze_excerpt(moving_video, regions, method='ic', **kwargs)
    alternate, other = analyze_excerpt(moving_video, regions, method='lk', **kwargs)
    assert primary['measurement_supported']
    check = compare_estimators(primary, traces, alternate, other)
    assert check['available'] and check['dominant_peaks_agree']
    assert abs(check['dominant_frequency_primary'] - 2.5) < .25


def test_duplicate_files_are_not_counted_as_independent_experiments(moving_video, tmp_path, monkeypatch):
    import shutil
    import vibrovision.video_lab as lab
    shutil.copyfile(moving_video, tmp_path / 'tres.mp4')
    shutil.copyfile(moving_video, tmp_path / 'cuatro.mp4')
    monkeypatch.setattr(lab, 'LISS_SHOWCASE_SHA256', lab.file_sha256(moving_video))
    inventory, tasks, pending = prepare_liss_inputs(tmp_path)
    assert inventory[1]['duplicate_of'] == 'tres.mp4'
    assert len(tasks) == 1 and not pending


def test_unknown_content_does_not_inherit_liss_regions(moving_video, tmp_path):
    import shutil
    shutil.copyfile(moving_video, tmp_path / 'tres.mp4')
    inventory, tasks, pending = prepare_liss_inputs(tmp_path, ['tres.mp4'])
    assert not tasks and pending[0]['status'] == 'unknown_content_profile'


def test_excerpt_does_not_silently_truncate(moving_video):
    with pytest.raises(ValueError, match='excede'):
        analyze_excerpt(moving_video, {'A': [45, 55, 26, 26], 'B': [150, 55, 26, 26]},
                        start_s=8, duration_s=10)
