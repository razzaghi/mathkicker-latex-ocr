from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex.util import process_raw_latex_code
from nougat_latex import NougatLaTexProcessor

app = Flask(__name__)

# Function to run Nougat LaTeX model
def run_nougat_latex(img_path, device="cpu"):
    # Initialize model and processor
    model = VisionEncoderDecoderModel.from_pretrained("Norm/nougat-latex-base").to(device)
    tokenizer = NougatTokenizerFast.from_pretrained("Norm/nougat-latex-base")
    latex_processor = NougatLaTexProcessor.from_pretrained("Norm/nougat-latex-base")

    # Load image
    image = Image.open(img_path)
    if not image.mode == "RGB":
        image = image.convert('RGB')

    pixel_values = latex_processor(image, return_tensors="pt").pixel_values
    task_prompt = tokenizer.bos_token
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False,
                                  return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=2,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    
    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
    sequence = process_raw_latex_code(sequence)

    return sequence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded image
        img_path = './uploaded_image.png'  # You can choose another path or handle the file differently
        file.save(img_path)

        # Run the Nougat LaTeX model inference
        result = run_nougat_latex(img_path)

        # Return result as JSON response
        return jsonify({'latex_code': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run with Hypercorn
if __name__ == '__main__':
    import hypercorn.asyncio
    from hypercorn.config import Config
    config = Config()
    hypercorn.asyncio.run(app, config)
