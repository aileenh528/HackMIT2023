import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from flask import Flask, request, render_template

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def generated():
    if request.method == 'POST':
        mood = request.form.get("mood")

        inputs = processor(
            text=[mood],
            padding=True,
            return_tensors="pt",
        )

        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

        return render_template('result.html')