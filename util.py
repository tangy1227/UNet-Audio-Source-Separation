import torch
import numpy as np
import musdb
import random
import json
import os
import librosa

from tqdm import tqdm
from splitter import Splitter
from museval import metrics
from datetime import datetime

import torchaudio.transforms as transforms
from torch import Tensor, nn
from torch.nn import functional as F

class utility():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_fft = 2048
        self.hop_length = 1024
        self.n_mels = 128
        self.sr = 44100
        self.mus = musdb.DB(
            root="/home/ytang363/7100_spr2023/musdb18",
            is_wav=False,
            subsets=["test"]
        )

    def load_model(self, modelPath, stem_names=["vocals","accompaniment"]):
        model = Splitter(stem_names=stem_names) #"accompaniment"
        state_dict = torch.load(modelPath, map_location=self.device)
        model.load_state_dict(state_dict['model_state_dict'],strict=False)
        return model
    
    def plot_output(predict, stft_input):
        mask_sum = sum([m**2 for m in predict.values()])
        mask_sum += 1e-10

        mask = (predict['vocals']**2 + 1e-10 / 2) / (mask_sum)
        mask = mask.transpose(2, 3)
        print(mask.shape)

        mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)
        mask = mask.squeeze(0)[:, :, :stft_input.size(2)].unsqueeze(-1) # 2 x F x L x 1
        print(mask.shape)

        # Plot masked instrumental
        stft_masked = stft_input * mask
        return stft_masked.cpu().detach().numpy()[0][:, :, 0]
    
    def inverse_stft(stft: Tensor) -> Tensor:
        """Inverses stft to wave form"""

        pad = 4096 // 2 + 1 - stft.size(1) # 4096 // 2 + 1 - 1024 = 1025
        print(stft.size())
        # stft = F.pad(stft, (0, 0, 0, 0, 0, pad)) # 2 x 2049 x 431 x 2
        stft = F.pad(stft, (0, 0, 0, pad))
        print(stft.size()) 

        wav = torch.istft(
            stft,
            4096,
            hop_length=1024,
            center=True,
            window=nn.Parameter(torch.hann_window(4096), requires_grad=False).to("cuda"),
        )
        return wav.detach()
    
    def cal_SDR(self, model, stems, duration: int = 180, niter: int = 50, shuffle: bool = False):
        for stem in stems:
            stem_sdr_list = []
            now_str = datetime.now().strftime("%m%d-%H%M")
            directory = '/home/ytang363/7100_spr2023/logs-sdr/{}-{}.json'.format(now_str,stem)

            if shuffle == True:
                niter = np.random.choice(50, size=niter, replace=False)
                iterator = tqdm(niter, desc="SDR")
            else:
                iterator = tqdm(range(niter), desc="SDR")

            for i in iterator:
                track = self.mus.tracks[i]

                duration = duration if track.duration > duration else np.floor(track.duration).astype(int)
                target = np.zeros((1,(duration-1)*44100, 2))
                est = np.zeros((1,(duration-1)*44100, 2))
                
                track.chunk_duration = track.duration
                track.chunk_start = np.floor(random.uniform(0, track.duration - duration))
                # print("{}: {}".format(i,track))

                ## Target ##
                wav_vocal = track.targets[stem].audio
                target[0] = wav_vocal[:(duration-1)*44100]

                ## Estimated ##
                wav_ = track.audio.T
                wav = torch.Tensor(wav_).to(self.device)
                with torch.no_grad():
                    stems = model.separate(wav)
                    vocal = stems[stem]
                    est_vocal = vocal.cpu().detach().numpy().T
                est[0] = est_vocal[:(duration-1)*44100]

                if np.all(target == 0) or np.all(est == 0):
                    # print("either reference or est has all zero array")
                    eps = 1e-8
                else:
                    eps = 0

                (sdr, _, sir, sar, perm) = metrics.bss_eval(target+eps, est+eps, 
                                                            window=np.inf, bsseval_sources_version=False)
                sdr = max(sdr, np.array([[0]]))
                # print("SDR: {}".format(sdr))
                stem_sdr_list.append(sdr)

            result = np.concatenate(stem_sdr_list)
            my_list = result.flatten().tolist()

            with open(directory, 'w') as f:
                json.dump(my_list, f)

        return result
    
    def mel_spec(self, wav_out, time_domain=True):
        if time_domain:
            output_mono = torch.mean(wav_out, dim=0, keepdim=True)
            output_mono = wav_out
        else:
            funct = Splitter(stem_names=["vocals"])
            output = funct.inverse_stft(wav_out)
            output_mono = torch.mean(output, dim=0, keepdim=True)
            output_mono = output
            # output_mono.requires_grad = True

        # mel_spectrogram = transforms.MelSpectrogram(sample_rate=self.sr, n_fft=self.n_fft, 
        #                                             hop_length=self.hop_length, n_mels=self.n_mels).to(self.device)
        # mel_spec = mel_spectrogram(output_mono).to(self.device)
        # mel_spec_db = transforms.AmplitudeToDB()(mel_spec)

        output_mono = output_mono.requires_grad_()
        output_mono = output_mono.cpu().detach().numpy()
        mel_spec = librosa.feature.melspectrogram(y=output_mono, sr=self.sr, n_fft=self.n_fft, 
                                                  hop_length=self.hop_length, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec = torch.tensor(mel_spec).requires_grad_()
        mel_spec_db = torch.tensor(mel_spec_db).requires_grad_()
        
        # return mel_spec.squeeze(0), mel_spec_db.squeeze(0)
        return mel_spec, mel_spec_db
    
    def mag2db(self, mag_spec, ref=1.0, eps=1e-10):
        # Convert magnitude spectrogram to decibels
        mag_spec = np.abs(mag_spec.cpu().detach().numpy())
        db_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)
        # db_spec = 20 * torch.log10(torch.abs(mag_spec) + eps / ref)
        db_spec_tensor = torch.from_numpy(db_spec).to(self.device).requires_grad_(True)
        return db_spec_tensor
    
if __name__ == "__main__":
    path = "/home/ytang363/7100_spr2023/model/20230418-05_ep-600_b-8.pt"
    util = utility()
    stem_names = ['vocals','accompaniment'] #['drums', 'bass', 'other', 'vocals']
    model = util.load_model(modelPath=path, stem_names=stem_names)
    sdr = util.cal_SDR(model=model,stems=stem_names,duration=120,niter=50,shuffle=False)
    print(sdr)
