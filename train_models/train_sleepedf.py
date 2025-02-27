from sklearn.model_selection import KFold, train_test_split
import os
import torch
import numpy as np
import argparse
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.data.sleepedf import get_dataloader
from src.models.eeg_models import EEGNetModel, calc_class_weight
from copy import deepcopy
import wandb

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    if args.model_type in ['SleepStagerEldele2021', 'ConvModel']:
        channels = ['EEG Fpz-Cz']
        balanced = False
        normalize = True
    else:
        channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
        balanced = True
        normalize = True 
    if device == 'cuda':
        num_workers = 10
    else:
        num_workers = 0
    channels = ['EEG Fpz-Cz']
    train_dataloader = get_dataloader(args.data_path, 'train', split = args.split, batch_size=args.batch_size, num_workers=num_workers, balanced = balanced, channels = channels, normalize=normalize)
    val_dataloader = get_dataloader(args.data_path, 'val', split = args.split, balanced = False, batch_size=args.batch_size, shuffle = False, num_workers=num_workers, channels = channels, normalize=normalize)
    test_dataloader = get_dataloader(args.data_path, 'test', split = args.split, balanced = False, batch_size=args.batch_size, shuffle = False, num_workers=num_workers, channels=channels, normalize=normalize)
    
    
    labels_count = np.unique(train_dataloader.dataset.labels, return_counts=True)[1]
    weights = calc_class_weight(labels_count)
    model = EEGNetModel(model_type=args.model_type, input_shape=(len(channels), 3000), lr=args.lr, weight_decay=args.weight_decay, class_weights=weights, device=device, conv_dropout=args.conv_dropout)
    logger = WandbLogger(project = 'explainability')
    logger.experiment.config.update(args)

    early_stop_callback = EarlyStopping(monitor="val/f1", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    # create checkpoint folder if it does not exist
    if not os.path.exists('models/sleepedf/checkpoints'):
        os.makedirs('models/sleepedf/checkpoints')
    checkpoint = ModelCheckpoint(monitor='val/f1', mode='max', dirpath = 'models/sleepedf/checkpoints')

    trainer = L.Trainer(max_epochs=args.n_epochs, accelerator = device, logger = logger, callbacks=[early_stop_callback, lr_logger, checkpoint])
    trainer.fit(model, train_dataloader, val_dataloader)
    if args.use_edf_20:
        output_path = f"{args.output_path}{args.model_type}_edf20_{args.split}.pt"
    else:
        output_path = f"{args.output_path}{args.model_type}_{args.split}.pt"
    best_model_path = checkpoint.best_model_path
    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    torch.save(model.state_dict(), output_path)
    
    
    acc = model.test_model(test_dataloader)
    wandb.run.summary["test_accuracy"] = acc[-1]
    wandb.run.summary["test_balanced_accuracy"] = acc[0]
    wandb.run.summary["test_f1"] = acc[1]

    logger.experiment.finish()
    print(f"Test accuracy: {acc}")    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEGNet on sleep data')
    parser.add_argument('--data_path', default = '/Users/theb/Desktop/data/sleep_edf/epoched', type=str, help='Path to epoched data')
    parser.add_argument('--use_edf_20', default = True, type=eval, help='Only use the first 20 subjects')
    parser.add_argument('--output_path', default = 'models/sleepedf/test.pt', type=str, help='Path to save model')
    parser.add_argument('--nsplits', type=int, default = 5, help='Number of splits for cross-validation')
    parser.add_argument('--split', type=int, default = 0, help='Number of splits for cross-validation')
    parser.add_argument('--batch_size', type=int, default = 128, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default = 1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default = 1e-3, help='Learning rate')
    parser.add_argument('--conv_dropout', type=float, default = 0.1, help='Dropout rate for AudioNet')
    parser.add_argument('--weight_decay', type=float, default = 0., help='Weight decay')
    parser.add_argument('--model_type', type=str, default = 'ConvModel', help='Model type')
    parser.add_argument('--patience', type=int, default = 3, help='Patience for early stopping')
    args = parser.parse_args()
    main(args)
