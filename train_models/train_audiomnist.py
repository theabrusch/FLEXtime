from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from src.models import AudioNet
from src.data import AudioNetDataset
from src.utils import EarlyStopping
import numpy as np

labeltype = 'gender'
splits = np.arange(1, 5)
for split in splits:
    print(f"Split {split}")
    data_path = '/Users/theb/Desktop/data/AudioMNIST/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    dset = AudioNetDataset(data_path, True, 'train', splits = split, labeltype = labeltype)
    val_dset = AudioNetDataset(data_path, True, 'validate', splits = split, labeltype = labeltype)
    dloader = DataLoader(dset, batch_size=100, shuffle=True)

    val_dloader = DataLoader(val_dset, batch_size=100, shuffle=True)
    test_dset = AudioNetDataset(data_path, True, 'test', splits = split, labeltype = labeltype, subsample=1000, seed=0)                     
    test_dloader = DataLoader(test_dset, batch_size=100, shuffle=True)

    if labeltype == 'digit':
        num_classes = 10
    else:
        num_classes = 2
    model = AudioNet(input_shape=(1, 8000), num_classes=num_classes).to(device)

    epochs = 50
    collect_loss = []
    initial_lr = 0.0001
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    model.train()
    earlystopping = EarlyStopping(patience = 7, verbose = True, path = 'models/checkpoint.pt')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dloader:
            optimizer.zero_grad()
            output = model(batch[0].float().to(device))
            loss = loss_fn(output, batch[1].long().to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
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
    torch.save(best_model.cpu().state_dict(), f'AudioNet_{labeltype}_{split}.pt')