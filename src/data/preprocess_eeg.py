import mne
import os
import glob
import argparse


def preprocess_EEG(file, out_folder = None):

    raw = mne.io.read_raw_edf(file, stim_channel='Event marker', preload = True)
    file_split = file.split('/')
    subject = file_split[-1][:5]
    session = file_split[-1][5:7]
    file_path = '/'.join(file_split[:-1])
    
    anno_path = glob.glob(f'{file_path}/{subject}{session}*-Hypnogram.edf')[0]
    
    annot_train = mne.read_annotations(anno_path)
    raw.set_annotations(annot_train, emit_warning=False)

    # Extract annotations for sleep stages
    annotations = raw.annotations
    # crop to 30 minutes before and after the first and last sleep stage
    annotations.crop(annotations[1]["onset"] - 30 * 60, annotations[-2]["onset"] + 30 * 60, use_orig_time=False)
    raw.set_annotations(annotations, emit_warning=False)

    # Convert annotations to numeric labels (example mapping)
    label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
    event_map = {v: v for v in label_map.values()}
    events = mne.events_from_annotations(raw, event_id=label_map, chunk_duration=30)[0]
    # create epoched data
    epochs = mne.Epochs(raw, events, tmin=0, tmax=30 - 1 / 100, preload=True, decim=1,
                    baseline=None, reject_by_annotation=True, verbose=False)
    
    out_path = f'{out_folder}{subject}/'
    os.makedirs(out_path, exist_ok = True)
    epochs.save(f'{out_path}{subject}{session}-epo.fif', overwrite = True)

def main(args):
    root_folder = args.root_folder + '*-PSG.edf'

    subjects = glob.glob(root_folder)
    ss = [subj.split('/')[-1][:5] for subj in subjects]
    file = subjects[0]

    file_split = file.split('/')
    subject = file_split[-1][:5]
    session = file_split[-1][5:7]
    file_path = '/'.join(file_split[:-1])

    for i, subject in enumerate(subjects):
        print('Subject', i+1, 'of', len(subjects))
        preprocess_EEG(subject, out_folder = args.out_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='/path/to/edf/')
    parser.add_argument('--out_folder', type=str, default='/path/to/save/fif/')
    args = parser.parse_args()
    main(args)