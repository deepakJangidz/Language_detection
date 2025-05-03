# import os
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from pydub import AudioSegment
# import numpy as np

# # Set paths
# input_root = 'Indian_Languages/hindi'  # e.g., './Indian_Language_Dataset   path_to_extracted_dataset'
# output_root = 'melspectrograms'

# # Ensure output directory exists
# os.makedirs(output_root, exist_ok=True)

# def convert_to_melspectrogram(mp3_path, output_path):
#     y, sr = librosa.load(mp3_path, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     S_DB = librosa.power_to_db(S, ref=np.max)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Walk through each language folder
# for root, dirs, files in os.walk(input_root):
#     for file in files:
#         if file.endswith(".mp3"):
#             lang_folder = os.path.relpath(root, input_root)
#             input_file = os.path.join(root, file)
#             output_dir = os.path.join(output_root, lang_folder)
#             os.makedirs(output_dir, exist_ok=True)

#             output_filename = os.path.splitext(file)[0] + ".png"
#             output_file = os.path.join(output_dir, output_filename)

#             try:
#                 convert_to_melspectrogram(input_file, output_file)
#                 print(f"Converted: {input_file} -> {output_file}")
#             except Exception as e:
#                 print(f"Failed to convert {input_file}: {e}")



import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
from re import findall
import re

base_path = "Indian_Languages"

def create_melspec(fold, train=20000, test=5000, n_fft=2048, hop_length=512, n_mels=64, f_min=20, f_max=8000, sample_rate=16000):
    print(f'Creating melspectrograms in {fold}')
    folder = os.path.join(base_path, fold)
    audio_files = [f for f in os.listdir(folder) if f.endswith(".mp3")]

    # define train/test folders based on language in folder name
    lang = fold  # assumes folder names like 'hindi_mp3'
    spectrogram_path_train = f"data/train/{lang}"
    spectrogram_path_test = f"data/test/{lang}"

    os.makedirs(spectrogram_path_train, exist_ok=True)
    os.makedirs(spectrogram_path_test, exist_ok=True)

    counter = 0
    for audio_file in audio_files:
        try:
            clip, sr = librosa.load(os.path.join(folder, audio_file), sr=None)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            break


        fig = plt.figure(figsize=[0.75, 0.75])
        ax = fig.add_subplot(111)
        ax.axis('off')

        if len(clip) >= 76000:
            clip = clip[16000:16000+60000]
        else:
            clip = clip[:60000]

        
        mel_spec = librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0, fmin=f_min, fmax=f_max)

        librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max),
                                 sr=sample_rate, hop_length=hop_length, fmax=f_max)

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

        counter += 1
        if counter >= train + test:
            print(f"Finished processing {fold}.")
            break

# scan for *_mp3 folders
# lang_folders = os.listdir("Indian_Languages")
# lang_folders = [f for f in all_folders if f.endswith("_mp3")]
lang_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]


print(f"The MP3 files are stored in the folders: {lang_folders}")

# process each folder
for folder in lang_folders:
    create_melspec(folder, train=20000, test=2000)
