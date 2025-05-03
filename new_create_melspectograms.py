import os
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np
from re import findall

# Set base path
base_path = "Indian_Languages"
output_base = "new_data"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_melspec(fold, train=20000, test=5000, n_fft=2048, hop_length=512, n_mels=64, f_min=20, f_max=8000, sample_rate=16000):
    print(f"Creating melspectrograms in {fold}")
    folder = os.path.join(base_path, fold)
    audio_files = [f for f in os.listdir(folder) if f.endswith(".mp3")]

    # Create output directories
    lang = fold  # assuming folder name is like 'hindi', 'tamil', etc.
    spectrogram_path_train = os.path.join(output_base, "train", lang)
    spectrogram_path_test = os.path.join(output_base, "test", lang)
    os.makedirs(spectrogram_path_train, exist_ok=True)
    os.makedirs(spectrogram_path_test, exist_ok=True)

    counter = 0
    for audio_file in audio_files:
        file_path = os.path.join(folder, audio_file)
        try:
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.to(device)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate).to(device)
                waveform = resampler(waveform)

            # Clip to fixed size
            if waveform.shape[1] >= 76000:
                waveform = waveform[:, 16000:16000 + 60000]
            else:
                waveform = waveform[:, :60000]

            # Compute Mel spectrogram
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=1.0
            ).to(device)(waveform)

            mel_spec_db = torchaudio.functional.amplitude_to_DB(
                mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
            )
            mel_spec_db = mel_spec_db.squeeze().cpu().numpy()

            # Plot and save
            fig = plt.figure(figsize=[0.75, 0.75])
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.imshow(mel_spec_db, origin='lower', aspect='auto', cmap='viridis')

            # Create filename
            speaker = findall(r'(?<=mp3_).*?[._-]', audio_file)
            if not speaker:
                speaker = ['spk']
            new_name = f"{speaker[0]}{counter}.png"

            if counter >= train:
                filename = os.path.join(spectrogram_path_test, new_name)
            else:
                filename = os.path.join(spectrogram_path_train, new_name)

            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print(f"Saved {filename}")
            counter += 1
            if counter >= train + test:
                print(f"Finished processing {fold}.")
                break

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

# List all folders in base path
lang_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
print(f"Detected language folders: {lang_folders}")

# Process each folder
for folder in lang_folders:
    create_melspec(folder, train=20000, test=2000)
