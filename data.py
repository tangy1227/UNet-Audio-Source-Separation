from typing import Optional, Union, Tuple, List, Any, Callable
import random

from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import tqdm

class musdbDataset(Dataset):
    def __init__(
        self,
        targets: List[str] = None, # ['drums', 'bass', 'other', 'vocals']
        is_train: bool = True,
        root: str = "/home/ytang363/7100_spr2023/musdb18",
        subsets: str = "train",
        split: str = "train",
        seq_duration: Optional[float] = 6.0,    # Duration for extraction
        samples_per_track: int = 64,            # sets the number of samples, yielded from each track per epoch
        seed: int = 27,                         # randomness of dataset iterations
    ) -> None:
        
        super().__init__()
        import musdb

        self.seed = seed
        random.seed(seed)
        self.seq_duration = seq_duration
        self.targets = [s for s in targets] if targets else ['vocals', 'accompaniment']
        self.subsets = subsets
        self.samples_per_track = samples_per_track
        self.mus = musdb.DB(
            root=root,
            is_wav=False,
            subsets=["train" if is_train else "test"],
            split=split
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __len__(self):
        # unmix_len = len(self.mus.tracks) * self.samples_per_track
        other_len = len(self.mus)
        return other_len

    def __getitem__(self, index):
        
        mus = self.mus
        track = mus.tracks[index]
        # track = mus.tracks[index // self.samples_per_track] # ind 1-64, get 64 samples from each track 
        # print(1 // samples_per_track)
        # print(track)

        """Other Method"""
        # targets = ['vocals']
        track.chunk_duration = self.seq_duration
        track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
        x_wav = torch.as_tensor(track.audio.T, dtype=torch.float32)

        y_target_wavs = {}
        for name in self.targets:
            y_target_wavs[name] = torch.tensor(track.targets[name].audio.T, dtype=torch.float32)
            # y_target_wavs[name] = torch.tensor(track.audio.T, dtype=torch.float32) # Change target the same with input

        return x_wav, y_target_wavs

        """Open Unmix Method
        audio_sources = []
        target_ind = None
        for k, source in enumerate(mus.setup['sources']):
            # k --> 0,1,2,3
            # source --> vocals, drums, bass, other
            if source == self.target:
                # Memorize index of target source
                target_ind = k
            
            # excerpt duration
            track.chunk_duration = self.seq_duration
            # random start position
            track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
            
            # Get seq_duration of the audio
            # track.sources[source] --> stem.mp4 file dir
            print("{}, source = {}".format(track.sources[source], source))
            audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
            audio_sources.append(audio)
        
        stems = torch.stack(audio_sources, dim=0)
        print(stems.shape)
        x = stems.sum(0)
        if target_ind is not None:
            print("target_ind: {}".format(target_ind))
            y = stems[target_ind]
        else:
            print(list(self.mus.setup["sources"].keys()))
        """


if __name__ == "__main__":
    # dataset = musdbDataset()
    # x_wav, y_target_wavs = dataset[0]
    # print(y_target_wavs)
    batch_size = 16

    train_dataset = musdbDataset()
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size = batch_size,
        drop_last=True
    )
    validation_dataset = musdbDataset(is_train=True, split="valid")
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, drop_last=True)
    print(len(train_dataloader))
    print(len(validation_loader))
    # # batch: {"instrument": Tensor[batch_size, channels, sample_len]}
    # for batch_index, batch in enumerate(train_dataloader):
    #     x_wav_batch, y_target_wavs_batch = batch
    #     if batch_index == 0:
    #         for i in range(batch_size):
    #             # print(batch[0][i].size())
    #             # print(batch[1]['vocals'][i].size())
    #             print(x_wav_batch)
    #     else:
    #         break


    """ # Open-Unmix Dataloader
    # import musdb

    # target = "vocals" # target to be separated
    # seq_duration = 3.0
    # root = "/Users/Owen/Desktop/MUSI7100_Spr2023/musdb18"
    # samples_per_track = 64
    # BATCH_SIZE = 1


    # # Get train dataset
    # train_dataset = MUSDBDataset(
    #     split="train",
    #     samples_per_track=args.samples_per_track,
    #     seq_duration=args.seq_dur,
    #     source_augmentations=source_augmentations,
    #     random_track_mix=True,
    #     **dataset_kwargs,
    # )

    # # Get validation dataset
    # valid_dataset = MUSDBDataset(
    #     split="valid", samples_per_track=1, seq_duration=None, **dataset_kwargs
    # )
    """