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

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import tempfile
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment

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

# App title
st.title("Language Detection AI")
st.markdown("AI bot trained to detect the language from speech using Deep Learning")

# 🎤 Voice recording class
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.recorded_frames.append(audio)
        return frame

# === SIDEBAR ===
st.sidebar.markdown("## Upload or Record Audio")
mp3_file = st.sidebar.file_uploader("Upload MP3 File:", type=["mp3"])
preset = st.sidebar.radio("Or Choose a Preset:", options=["None"] + classes)

st.sidebar.markdown("## Or Record Your Voice")
# ctx = webrtc_streamer(
#     key="recorder",
#     mode="sendonly",
#     in_audio=True,
#     audio_processor_factory=AudioProcessor
# )
ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,  
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

recorded_file_path = None
if ctx and ctx.audio_processor and ctx.audio_processor.recorded_frames:
    st.sidebar.success("Recording complete. Processing...")
    recorded_audio = np.concatenate(ctx.audio_processor.recorded_frames, axis=1).flatten().astype(np.float32)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        sf.write(tmp_wav.name, recorded_audio, samplerate=16000)
        recorded_file_path = tmp_wav.name

    # Apply noise reduction
    reduced_audio = nr.reduce_noise(y=recorded_audio, sr=16000)
    clean_path = recorded_file_path.replace(".wav", "_clean.wav")
    sf.write(clean_path, reduced_audio, 16000)

    # Convert to MP3
    sound = AudioSegment.from_wav(clean_path)
    mp3_path = clean_path.replace(".wav", ".mp3")
    sound.export(mp3_path, format="mp3")

    mp3_file = mp3_path
    st.sidebar.success("Cleaned voice ready for detection.")

# === MAIN LOGIC ===
imagewidth = 550
placeholder = st.empty()
placeholder.image('assets/speechbubble.png', width=imagewidth)

# Use preset audio if nothing is uploaded
if mp3_file is None and preset in classes:
    mp3_file = f'assets/{preset}_preset.mp3'

if mp3_file is not None:
    placeholder.image("assets/speechbubble2.png", width=imagewidth)
    st.audio(mp3_file)

    status_text = st.empty()
    progress_bar = st.empty()
    predictions = []

    status_text.text('Press Start To Begin Detection...')
    if st.button('Start'):
        status_text.text('Rendering Audio File...')
        clip, sample_rate = load_audio(mp3_file, sr=16000)

        # Process only the first 60 seconds of the audio
        duration = len(clip)
        end = min(duration, 60 * sample_rate)  # 60 seconds in samples

        # Make sure to only use the first 60 seconds
        clip_new = clip[:end]

        # Convert the clip to mel spectrogram
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

        # Run the model on the spectrogram image
        model.eval()
        output = model(image)
        _, predicted = torch.max(output, dim=1)
        predictions.append(classes[predicted[0].item()])

        status_text.empty()
        progress_bar.empty()

        # Calculate percentage breakdown of predictions
        results = Counter(predictions)
        for c in classes:
            results.setdefault(c, 0)

        df = pd.DataFrame.from_dict(results, orient='index', columns=['percent']).T.div(1) * 100
        highest_prediction = df.idxmax(axis=1).iloc[0]
        confidence = df.loc['percent', highest_prediction]

        language_image_map = {
            "Bengali": "Group 300.png",
            "Gujarati": "Group 301.png",
            "Hindi": "Group 299.png",
            "Kannada": "Group 302.png",
            "Malayalam": "Group 303.png",
            "Marathi": "Group 304.png",
            "Punjabi": "Group 305.png",
            "Tamil": "Group 306.png",
            "Telugu": "Group 307.png",
            "Urdu": "Group 308.png"
        }

        if confidence >= 60:
            image_filename = language_image_map.get(highest_prediction, "speechbubble_default.png")
        else:
            image_filename = "speechbubble_uncertain.png"

        placeholder.image(f'assets/{image_filename}', width=imagewidth)

        st.markdown("----------")
        st.markdown("### Breakdown by percentage")
        st.dataframe(df, width=600)
        st.bar_chart(df.T, height=400, use_container_width=True)
