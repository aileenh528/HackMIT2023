import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from flask import Flask, request, render_template
import requests
import datetime

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
        fitbit_userid = request.form.get("fitbit-userid")
        print(fitbit_userid)

        if fitbit_userid is not None and fitbit_userid != "":
            try:
                user_id = fitbit_userid
                today = datetime.date.today()
                yesterday = today - datetime.timedelta(days=1)
                url = f"https://api.tryterra.co/v2/sleep?user_id={user_id}&start_date={yesterday.isoformat()}&end_date={today.isoformat()}&to_webhook=true&with_samples=true"

                headers = {
                    "accept": "application/json",
                    "dev-id": "hackmit-testing-RoRdxSOdgz",
                    "x-api-key": "2r9NCMIz-90cLUyvANxGZqLkEdBXJXbF"
                }

                response = requests.get(url, headers=headers)
                print(response)

                sleep_efficiency = response.json()["data"]["sleep_durations_data"]["sleep_efficiency"]

                if sleep_efficiency < 80:
                    mood += f"My sleep efficiency was low, at {sleep_efficiency}%, so I'm not feeling well-rested."
                elif sleep_efficiency < 95:
                    mood += f"My sleep efficiency was fine, at {sleep_efficiency}%, so I'm feeling okay."
                else:
                    mood += f"My sleep efficiency was great, at {sleep_efficiency}%, so I'm feeling very well-rested."
            except:
                pass

        print(mood)
        inputs = processor(
            text=[mood],
            padding=True,
            return_tensors="pt",
        )

        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write("static/musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

        return render_template('result.html')

if __name__ == "__main__":
    app.run()
