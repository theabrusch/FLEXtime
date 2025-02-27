from src.data import generate_time_series_voigt, generate_time_series_voigt_noisy, divide_frequency_axis, generate_time_series_width
from src.models import ConvModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from src.utils import EarlyStopping
import numpy as np

X = 32
Fs = 1000
length = 2000
noise_level = 0.1
num_sample_regions = 10
num_frequencies = 20
gamma = 1.5
scale = 1
sigma = 0.5
dataset = 'synth_width'
all_regions = divide_frequency_axis(Fs, num_regions=X)

# Select 4 regions to be the salient regions (you can select them randomly or manually)
salient_region_indices = [6, 13, 20, 27]
salient_regions = [all_regions[i] for i in salient_region_indices]

splits = np.arange(5)
lengths = [2000]
for split in splits:
    for length in lengths:
        if dataset == 'synth_voigt':
            test_data, test_labels = generate_time_series_voigt(num_samples_per_class=1000//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                salient_regions=salient_regions,
                                                                all_regions=all_regions, sampling_frequency=Fs, 
                                                                duration=length/Fs, gamma=gamma, sigma = sigma, noise_level=noise_level, seed = None)
            train_data, train_labels = generate_time_series_voigt(num_samples_per_class=10**4//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                    salient_regions=salient_regions,
                                                                    all_regions=all_regions, sampling_frequency=Fs,
                                                                    duration=length/Fs, gamma=gamma, sigma = sigma, noise_level=noise_level, seed = None)
            val_data, val_labels = generate_time_series_voigt(num_samples_per_class=1000//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                salient_regions=salient_regions,
                                                                all_regions=all_regions, sampling_frequency=Fs,
                                                                duration=length/Fs, gamma=gamma, sigma = sigma, noise_level=noise_level, seed = None)
        elif dataset == 'synth_voigt_noisy':
            test_data, test_labels = generate_time_series_voigt_noisy(num_samples_per_class=1000//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                salient_regions=salient_regions,
                                                                all_regions=all_regions, sampling_frequency=Fs, 
                                                                duration=length/Fs, gamma=gamma, sigma=sigma, noise_level=noise_level, seed = None, scale = scale)
            train_data, train_labels = generate_time_series_voigt_noisy(num_samples_per_class=10**4//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                    salient_regions=salient_regions,
                                                                    all_regions=all_regions, sampling_frequency=Fs,
                                                                    duration=length/Fs, gamma=gamma, sigma=sigma, noise_level=noise_level, seed = None, scale = scale)
            val_data, val_labels = generate_time_series_voigt_noisy(num_samples_per_class=1000//16, num_sample_regions = num_sample_regions, num_frequencies=num_frequencies,
                                                                salient_regions=salient_regions,
                                                                all_regions=all_regions, sampling_frequency=Fs,
                                                                duration=length/Fs, gamma=gamma, sigma=sigma, noise_level=noise_level, seed = None, scale = scale)
        elif dataset == 'synth_width':
            train_data, train_labels, train_class_bands = generate_time_series_width(10**4, length/Fs, Fs, noise_level=noise_level)
            val_data, val_labels, val_class_bands = generate_time_series_width(1000, length/Fs, Fs, noise_level=noise_level)
            test_data, test_labels, test_class_bands = generate_time_series_width(1000, length/Fs, Fs, noise_level=noise_level)

        dloader = DataLoader(TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).long()), 
                                batch_size = 256, shuffle = True)
        val_dloader = DataLoader(TensorDataset(torch.tensor(val_data).float(), torch.tensor(val_labels)), batch_size=200)
        test_dloader = DataLoader(TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels)), batch_size=200)

        if dataset == 'synth_width':
            layers = 2
            hidden_size = 64
            kernel_size = 31
            model = ConvModel(output_size=3, hidden_layers = layers, hidden_size = hidden_size, kernel_size=kernel_size, adapt_avg = True, input_length = length)
            initial_lr = 5e-4
            weight_decay = 1e-6
        else:
            model = ConvModel(output_size=16, hidden_layers = 1, hidden_size = 64, kernel_size=31)
            initial_lr = 1e-4
            weight_decay = 0.
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        model.to(device)

        epochs = 100
        collect_loss = []
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

        earlystopping = EarlyStopping(patience = 10, verbose = True, path = 'models/checkpoint.pt', delta = 1e-6)
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in dloader:
                optimizer.zero_grad()
                output = model(batch[0].float().to(device))
                loss = loss_fn(output, batch[1].long().to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {epoch_loss/len(dloader)}")
            collect_loss.append(epoch_loss/len(dloader))
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch in val_dloader:
                    output = model(batch[0].float().to(device))
                    _, predicted = torch.max(output, 1)
                    total += batch[1].size(0)
                    correct += (predicted == batch[1].to(device)).sum().item()
                print(f"Validation Accuracy: {100 * correct / total}")
            earlystopping(-correct / total, model)
            if earlystopping.early_stop:
                print("Early stopping")
                break
            
        best_model = model
        best_model.load_state_dict(torch.load('models/checkpoint.pt'))
        best_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in test_dloader:
                output = best_model(batch[0].float().to(device))
                _, predicted = torch.max(output, 1)
                total += batch[1].size(0)
                correct += (predicted == batch[1].to(device)).sum().item()
            print(f"Test Accuracy: {100 * correct / total}")
            torch.save(best_model.cpu().state_dict(), f'models/synth_voigt/{dataset}_{noise_level}_{length}_{split}.pt')