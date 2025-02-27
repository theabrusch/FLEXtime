import os
import mne
from pathlib import Path

# preprocess edf files
path = '/work3/theb/timeseries/sleep_edf/'
output_path = '/work3/theb/timeseries/sleep_edf_epoched/'
subjs = os.listdir(path)
subjs = [s for s in subjs if not s.startswith('.')]
for subj in subjs:
    print(subj)
    files = os.listdir(f'{path}/{subj}')
    files = [f for f in files if not f.startswith('.')]
    # create output directory for subject
    Path(f'{output_path}/{subj}').mkdir(parents=True, exist_ok=True)
    for file in files:
        print(file)
        fif_file = f'{path}/{subj}/{file}'
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

        # Extract annotations for sleep stages
        annotations = raw.annotations
        annotations.crop(annotations[1]["onset"] - 30 * 60, annotations[-2]["onset"] + 30 * 60, use_orig_time=False)
        raw.set_annotations(annotations, emit_warning=False)

        # Convert annotations to numeric labels (example mapping)
        label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
        event_map = {v: v for v in label_map.values()}
        events = mne.events_from_annotations(raw, event_id=label_map, chunk_duration=30)[0]
        # create epoched data
        epochs = mne.Epochs(raw, events, tmin=0, tmax=30 - 1 / 100, preload=True, decim=1,
                        baseline=None, reject_by_annotation=True, verbose=False)
        file = file.replace('_raw.fif', '-epo.fif')
        epochs.save(f'{output_path}/{subj}/{file}', overwrite=True, verbose=False)
