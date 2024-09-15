import os
import torch
import time
import soundfile as sf
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("config.json")
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_dir="./")
XTTS_MODEL.eval()
if torch.cuda.is_available():
    XTTS_MODEL.cuda()


def text_to_wave(text, ref_path):
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=ref_path,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )
    out_wav = XTTS_MODEL.inference(
        text=text,
        language="vi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
    )
    return out_wav["wav"]

def text_to_speech(text, ref_path):
    try:
        waveform = text_to_wave(text, ref_path)
        if waveform is not None:
            # Save the waveform to a .wav file
            output_path = 'output.wav'
            sf.write(output_path, waveform, 24000)
            return output_path
        else:
            return None
    except Exception as e:
        print(f"Error in text_to_wav: {e}")
        return None


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure the device is set correctly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from flask import request, send_file, jsonify
import os

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        # Get the text from the request
        text = request.form.get('text', '')
        file = request.files.get('file')

        # Check if both text and file are provided
        if not text or not file:
            return jsonify({"error": "Missing 'text' or 'file' in request."}), 400

        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Process the audio file (here you can handle the audio as needed)
        print(f"File saved at: {file_path}")
        
        # Generate .wav file from the text (you need to adjust this part to suit your TTS logic)
        wav_file = text_to_speech(text, file_path)
        if wav_file is None:
            return jsonify({"error": "Error generating speech."}), 500

        # Send the generated .wav file as a response for download
        return send_file(wav_file, as_attachment=True)

    except Exception as e:
        print(f"Error in /synthesize: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port = 5000)
