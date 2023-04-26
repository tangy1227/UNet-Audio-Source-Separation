import torch
from pathlib import Path
from torchinfo import summary 

def test() -> None:
    # from unet import UNet, UNet2
    from unet_copy import UNet
    batch_size = 1
    n_channels = 2
    
    x = torch.randn(batch_size, n_channels, 512, 128)
    print(x.shape)
    net = UNet(in_channels=n_channels)
    y = net.forward(x)
    print(y.shape)
    new_y = y.squeeze(0)[:, :, :].unsqueeze(-1)
    print(new_y.shape)
    # summary(net)

def print_model() -> None:
    from splitter import Splitter

    # Load the saved model
    model = Splitter(stem_names=["vocals"])
    model.load_state_dict(torch.load('7100_spr2023/model/ep-200_b-16.pt'))
    for param in model.parameters():
        print(param.data)


def split(
    model_path: str = "/home/ytang363/7100_spr2023/model/20230424-16_ep-500_b-16.pt",
    input: str = "/home/ytang363/7100_spr2023/audio/All Souls Moon Mixture.wav",
    output_dir: str = "/home/ytang363/7100_spr2023/audio/output",
    offset: float = 30,
    duration: float = 10
) -> None:
    import librosa
    import soundfile

    from splitter import Splitter

    sr = 44100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # splitter = Splitter.from_pretrained(model_path).to(device).eval()
    model = Splitter(stem_names=['drums', 'bass', 'other', 'vocals']) #"accompaniment"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    print("Last Loss:", state_dict['loss'])

    # load wav audio
    fpath_src = input
    wav, _ = librosa.load(
        fpath_src,
        mono=False,
        res_type="kaiser_fast",
        sr=sr,
        duration=duration,
        offset=offset,
    )
    wav = torch.Tensor(wav).to(device)
    print(wav.shape)

    with torch.no_grad():
        stems = model.separate(wav)
        print(stems)
        vocal = stems["drums"]
        fpath_dst = Path(output_dir) / "test6.wav"
        soundfile.write(fpath_dst, vocal.cpu().detach().numpy().T, sr, "PCM_16")


if __name__ == "__main__":
    split()
    # test()