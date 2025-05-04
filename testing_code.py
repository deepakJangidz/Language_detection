import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchvision.transforms as tf
import io
from PIL import Image
from collections import Counter
from model import CNN_model_3

import av
import tempfile
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit config
st.set_page_config(page_title='Language AI', initial_sidebar_state='expanded')

# Class labels
classes = ["Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu"]
transformer = tf.Compose([tf.Resize([64, 64]), tf.ToTensor()])

@st.cache_resource
def load_model(path="cnn_model_trained_new.pt"):
    model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

@st.cache_resource
def load_audio(file, sr=16000):
    clip, sample_rate = librosa.load(file, sr=sr)
    return clip, sample_rate

model = load_model()

clip, sample_rate = load_audio('Indian_Languages/Bengali/0.mp3', sr=16000)

duration = len(clip)
num_samples = int(duration / 60000)
start = 0
end = 60000

model.eval()
for i in range(num_samples):
    prog = int(((i + 1) / num_samples) * 100)

    clip_new = clip[start:end]

    fig = plt.figure(figsize=[0.75, 0.75])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    mel_spec = librosa.feature.melspectrogram(
        y=clip_new, n_fft=2048, hop_length=512, n_mels=64,
        sr=sample_rate, power=1.0, fmin=20, fmax=8000
    )
    librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max), fmax=8000, sr=sample_rate)

    mel = io.BytesIO()
    plt.savefig(mel, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    image = Image.open(mel).convert('RGB')
    image = transformer(image).float().unsqueeze(0).to(device)

    actual_image= Image.open('data/train/Bengali/spk0.png')


    if isinstance(image, torch.Tensor):
        # If image is a tensor, convert it to a PIL Image first
        image = transforms.ToPILImage()(image)

    # Then apply the transformation
    generated_image_tensor = transformer(image).float().unsqueeze(0).to(device)
    actual_image_tensor = transformer(actual_image).float()

    mse = torch.mean((generated_image_tensor - actual_image_tensor) ** 2).item()
    st.write(f"Mean Squared Error between generated and actual image: {mse:.6f}")
