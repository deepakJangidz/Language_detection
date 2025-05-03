import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchvision
import torchvision.transforms as tf
import io
from PIL import Image
from collections import Counter

from model import CNN_model_3

############ SETUP ENVIRONMENT ################################################
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.beta_set_page_config(page_title='Language AI', initial_sidebar_state='expanded')
st.set_page_config(page_title='Language AI', initial_sidebar_state='expanded')


# Our labels for the classes (DO NOT CHANGE ORDER)
# classes = ["English", "French", "German", "Italian", "Spanish"]
classes = [ "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu"]

# transformations for our spectrograms
transformer = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])

# load our saved model function with caching
# @st.cache(allow_output_mutation=True)
# def load_model(path="trained_model_3_state.pt"):
@st.cache_resource
def load_model(path="cnn_model_trained.pt"):
    model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
@st.cache_resource
def load_audio(file, sr=16000):
    clip, sample_rate = librosa.load(file, sr=sr)
    return clip, sample_rate

# load the model
model = load_model("cnn_model_trained.pt")

"""
# Language Detection AI

AI bot trained to detect the language from speech using Deep Learning

##
"""
# landing screen
imagewidth = 550
placeholder = st.empty()
placeholder.image('assets/speechbubble.png', width=imagewidth)


############ SIDEBAR CONFIGURE ################################################

# Ask user to upload a voice file for language classification
st.sidebar.markdown('## Upload Voice Clip')
mp3_file = st.sidebar.file_uploader("Upload Your MP3 File Here:", type=["mp3"])

# allow users to choose a preloaded sample
st.sidebar.markdown("""## Or Use a Preset Audio Clip:""")
preset = st.sidebar.radio("Choose a Language", options=["None"]+classes)

st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space

st.sidebar.markdown(
"""
-----------
## Instructions
1. Upload a voice clip or select a **preset sample** audio file above
2. Click on 'Start' to begin detecting the language
3. View the results
4. Upload a new file or choose another preset to try again
5. If you are getting 'NaN', use clips longer than 4 seconds
""")

st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space
st.sidebar.header("") # add some empty space

st.sidebar.markdown(
"""
-----------

## Github
Creator: Deepak Kumar, Akanksha Bhayekar

Project Repository: [Link Here](https://github.com/deepakJangidz/Audio-Language-Classifier)
""")


############ RUN MODEL AND RETURN OUTPUT########################################
# if no files are uploaded, use preset ones by default

if mp3_file is None:
    if preset in classes:
        mp3_file = f'assets/{preset}_preset.mp3'
    else:
        mp3_file = None  # or handle unexpected preset

if mp3_file is not None:

    placeholder.image("assets/speechbubble2.png", width=imagewidth)
    st.audio(mp3_file) # allows users to play back uploaded files

    # set up the progress animations and initialize empty lists
    status_text = st.empty()
    progress_bar = st.empty()
    predictions = []

    status_text.text('Press Start To Begin Detection...')
    if st.button('Start'):

        # load our audio file into array
        status_text.text('Rendering Audio File...')
        clip, sample_rate = load_audio(mp3_file, sr=16000)

        duration = len(clip)
        num_samples = int(duration/60000) # number of samples we can extract from this file
        start = 0     # starting sample window
        end = 60000   # end sample window

        # take a sample from our uploaded voice clip
        model.eval()
        for i in range(num_samples):
            prog = int(((i+1)/num_samples)*100)
            status_text.text(f"Analysing Audio: {prog}%")
            progress_bar.progress(prog)

            clip_new = clip[start:end]

            # initialize our plot for the melspectrogram
            fig = plt.figure(figsize=[0.75,0.75])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            mel_spec = librosa.feature.melspectrogram(y=clip_new, n_fft=2048, hop_length=512, n_mels=64,
                                              sr=sample_rate, power=1.0, fmin=20, fmax=8000)
            librosa.display.specshow(librosa.amplitude_to_db(mel_spec, ref=np.max), fmax=8000, sr=sample_rate)

            mel = io.BytesIO()
            plt.savefig(mel, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            # load image to tensor and correct dimensions for our model
            image = Image.open(mel).convert('RGB')
            image = transformer(image).float()
            # image = torch.tensor(image, requires_grad=True)
            image = image.unsqueeze(0)

            # run predictions
            model.eval()
            output = model(image)
            print(f'output is {output}')
            _, predicted = torch.max(output, dim=1)
            print(f'predicted is {predicted}')


            # record predictions and update sample windows
            predictions.append(classes[predicted[0].item()])
            start += 60000
            end += 60000

        # output our results now
        status_text.empty()
        progress_bar.empty()

        # tally up the predictions for each sample
        results = Counter(predictions)
        print(f'Predictions are {predictions}')
        print(f'Result is {results}')

        # placeholder value of 0 for languages that did not appear
        for c in classes:
            if c in results.keys():
                pass
            else:
                results[c] = 0


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

        # Check the highest prediction and its confidence
        threshold = 60  # percentage threshold
        df = pd.DataFrame.from_dict(results, orient='index', columns=['percent']).T.div(num_samples) * 100
        highest_prediction = df.idxmax(axis=1).iloc[0]
        print(f'Highest prediction is {highest_prediction}')
        confidence = df.loc['percent', highest_prediction]
        print(f'Confidence is {confidence}')

       # Show image based on prediction confidence
        if confidence >= threshold:
            # Get the corresponding image filename from the map
            image_filename = language_image_map.get(highest_prediction, "speechbubble_default.png")
        else:
            # Show an uncertain image if the confidence is below threshold
            image_filename = "speechbubble_uncertain.png"

        # Display the chosen image
        placeholder.image(f'assets/{image_filename}', width=imagewidth)

        # Display breakdown of predictions
        st.write("""
        ----------
        # Breakdown
        By percentage of languages the AI thinks it is
        """)
        st.dataframe(df, width=600)
        st.bar_chart(df.T, height=400, use_container_width=True)
