# Audio Source Separation Based on U-Net
### [project slides](https://docs.google.com/presentation/d/1sz7kjI7p-CKNKelTpdJ9_IE-BO1JTE9s3UibCaeN2Gk/edit?usp=sharing)

## [Spleeter](https://github.com/deezer/spleeter) Baseline
U-Net based audio source separation by Deezer

## Modification
Experiment with different loss functions to compare their performance. Different spectro losses: mel-spectrogram and magnitude. Different distance costs: L1 and MSE.

## Files Note:
`logs` contains training loss log with tensorboard

`data.py` dataloader, experimented with both original baseline [Spleeter dataloader](https://github.com/deezer/spleeter) and [Open-Unmix dataloader](https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/data.py) dataloader

`display_mask.ipynb` contains what model outputs (a ratio mask) and spectrogram comparison between ground truth audio source (vocal) with separated audio source (mixture * ratio mask)

`run.py` model training code

`splitter.py` inferencing code


