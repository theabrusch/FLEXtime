# FLEXtime: Filterbank learning to explain time series

This repository implements FLEXtime (Filterbank Learning to EXplain time series).

Link to paper: https://arxiv.org/abs/2411.05841.

## Downloading and preprocessing the data
### AudioMNIST
### SleepEDF
Initially, download the Sleep Cassette data (https://www.physionet.org/content/sleep-edfx/1.0.0/). 
Run the following lin with correct root paths and desired output folders to preprocess the data to follow the correct format:
```
python3 preprocess_sleepedf.py --root_folder /path/to/sleep-casette/ --out_folder /path/to/sleepedf/
```

If you use this code, please cite the original paper:
```
@article{brusch2024flextime,
  title={FLEXtime: Filterbank learning for explaining time series},
  author={Br{\"u}sch, Thea and Wickstr{\o}m, Kristoffer K and Schmidt, Mikkel N and Jenssen, Robert and Alstr{\o}m, Tommy S},
  journal={arXiv preprint arXiv:2411.05841},
  year={2024}
}
```
