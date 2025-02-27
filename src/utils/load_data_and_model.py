from src.models import TransformerMVTS, AudioNet, ConvModel, EEGNetModel
from src.data import AudioNetDataset, process_Epilepsy, EpiDataset, process_PAM, PAMDataset, process_MITECG, generate_time_series_voigt, divide_frequency_axis, generate_time_series_voigt_noisy, EEGSleepDataset, FaultDetectionDataset
from torch.utils.data import DataLoader, TensorDataset
import torch

def load_data_and_model(args, device, subsample=False, seed = 0):
    if args.dataset == 'audio':
        if args.validate:
            test_dset = AudioNetDataset(args.data_path, True, 'validate', splits = args.split, labeltype = args.labeltype, subsample = args.n_samples, seed = seed)
        else:
            test_dset = AudioNetDataset(args.data_path, True, 'test', splits = args.split, labeltype = args.labeltype, subsample = args.n_samples, seed = seed)
        num_classes = 2 if args.labeltype == 'gender' else 10
        model_path = f'models/audio/AudioNet_{args.labeltype}_{args.split}.pt'
        model = AudioNet(input_shape=(1, 8000), num_classes=num_classes).eval()
        Fs = 8000
        time_dim = -1
        time_length = 8000
    elif args.dataset == 'epilepsy':
        if args.validate:
            _, test, _ = process_Epilepsy(split_no = args.split, device = 'cpu', base_path = args.data_path, n_samples=args.n_samples)
        else:
            _, _, test = process_Epilepsy(split_no = args.split, device = 'cpu', base_path = args.data_path, n_samples=args.n_samples)
        model_path = f'models/epilepsy/transformer_split={args.split}_cpu.pt'
        test_dset = EpiDataset(test.X, test.time, test.y)
        model = TransformerMVTS(
                d_inp = 1,
                max_len = 178,
                n_classes = 2,
                nlayers = 1,
                trans_dim_feedforward = 16,
                trans_dropout = 0.1,
                d_pe = 16,
                norm_embedding = False,
                device = device
            )
        Fs = 178
        time_dim = 1
        time_length = test.X.shape[0]
    elif args.dataset == 'pam':
        if args.validate:
            _, test, _ = process_PAM(split_no = args.split, device = 'cpu', base_path = args.data_path, gethalf = True, n_samples=args.n_samples)
        else:
            _, _, test = process_PAM(split_no = args.split, device = 'cpu', base_path = args.data_path, gethalf = True, n_samples=args.n_samples)
        if subsample:
            test.X = test.X[:, :2, :]
            test.time = test.time[:, :2]
            test.y = test.y[:2]
        test_dset = PAMDataset(test.X, test.time, test.y)
        model_path = f'models/pam/transformer_split={args.split}.pt'
        model = TransformerMVTS(
                d_inp = test.X.shape[2],
                max_len = test.X.shape[0],
                n_classes = 8,
                device = device
            )
        Fs = 100
        time_dim = 1
        time_length = test.X.shape[0]
    elif args.dataset == 'ecg':
        if args.validate:
            _, test, _, _ = process_MITECG(split_no = args.split, device = 'cpu', hard_split = True, normalize = False, 
                                        balance_classes = False, div_time = False, need_binarize = True, 
                                        exclude_pac_pvc = True, base_path = args.data_path, n_samples=args.n_samples)
        else:
            _, _, test, _ = process_MITECG(split_no = args.split, device = 'cpu', hard_split = True, normalize = False, 
                                            balance_classes = False, div_time = False, need_binarize = True, 
                                            exclude_pac_pvc = True, base_path = args.data_path, n_samples=args.n_samples)
        test_dset = EpiDataset(test.X, test.time, test.y)
        model_path = f'models/ecg/transformer_exc_split={args.split}.pt'
        model = TransformerMVTS(
                d_inp = test.X.shape[-1],
                max_len = test.X.shape[0],
                n_classes = 2,
                nlayers = 1,
                nhead = 1,
                trans_dim_feedforward = 64,
                trans_dropout = 0.1,
                #enc_dropout = 0.1,
                d_pe = 16,
                stronger_clf_head = False,
                pre_agg_transform = False,
                # aggreg = 'mean',
                norm_embedding = True,
                device = device
            )
        Fs = 360
        time_dim = 1
        time_length = test.X.shape[0]
    elif args.dataset == 'synth_voigt':
        # Divide the frequency axis into X regions (e.g., 10 regions)
        X = 32
        Fs = 1000
        duration = args.synth_length//Fs
        all_regions = divide_frequency_axis(Fs, num_regions=X)

        # Select 4 regions to be the salient regions (you can select them randomly or manually)
        salient_region_indices = [6, 13, 20, 27]
        salient_regions = [all_regions[i] for i in salient_region_indices]

        data, labels = generate_time_series_voigt(num_samples_per_class=args.n_samples//16, num_sample_regions = 10, num_frequencies=20,
                                                salient_regions=salient_regions,
                                                all_regions=all_regions, sampling_frequency=Fs, 
                                                duration=duration, gamma=1, sigma = 0.5, noise_level=0.2, seed = 0 + 1000*args.split)
        if args.synth_length == 2000:
            model_path = f'models/synth_voigt/synth_voigt_{args.split}.pt'
        else:
            model_path = f'models/synth_voigt/synth_voigt_{args.synth_length}.pt'
        test_dset = TensorDataset(torch.tensor(data).float(), torch.tensor(labels).long())
        model = ConvModel(output_size=16, hidden_layers = 1, hidden_size = 64, kernel_size=31)
        time_dim = -1
        time_length = args.synth_length
    elif args.dataset == 'synth_voigt_noisy':
        X = 32
        Fs = 1000
        duration = args.synth_length//Fs
        all_regions = divide_frequency_axis(Fs, num_regions=X)

        salient_region_indices = [6, 13, 20, 27]
        salient_regions = [all_regions[i] for i in salient_region_indices]

        data, labels = generate_time_series_voigt_noisy(num_samples_per_class=args.n_samples//16, num_sample_regions = 10, num_frequencies=20,
                                                salient_regions=salient_regions,
                                                all_regions=all_regions, sampling_frequency=Fs, 
                                                duration=duration, gamma=1.5, sigma = 0.5, noise_level=args.noise_level, seed = 0 + 1000*args.split, scale = 1)
        if args.synth_length == 2000:
            model_path = f'models/synth_voigt/synth_voigt_noisy_{args.noise_level}_{args.split}.pt'
        else:
            model_path = f'models/synth_voigt/synth_voigt_noisy_{args.noise_level}_{args.synth_length}.pt'
        test_dset = TensorDataset(torch.tensor(data).float(), torch.tensor(labels).long())
        model = ConvModel(output_size=16, hidden_layers = 1, hidden_size = 64, kernel_size=31)
        time_dim = -1
        time_length = args.synth_length
    elif args.dataset == 'sleepedf':
        if args.validate:
            test_dset = EEGSleepDataset(args.data_path, split = args.split, dataset = 'val', use_edf_20 = True, normalize = True, n_samples = args.n_samples, channels = ['EEG Fpz-Cz'])
        else:
            test_dset = EEGSleepDataset(args.data_path, split = args.split, dataset = 'test', use_edf_20 = True, normalize = True, n_samples = args.n_samples, channels = ['EEG Fpz-Cz'])
        model_path = f'models/sleepedf/1chan_AudioNet_edf20_{args.split}.pt'
        model = EEGNetModel(model_type='AudioNet', input_shape = (1, 3000), num_classes = 5, conv_dropout=0.5, device = device)
        Fs = 100
        time_dim = -1
        time_length = 3000
    else:
        raise ValueError('Dataset not supported')
    if not args.random_model:
        model.load_state_dict(torch.load(model_path, map_location=device))
    test_dloader = DataLoader(test_dset, batch_size=10, shuffle=False)
    model.eval().to(device)
    return test_dloader, model, Fs, time_dim, time_length

