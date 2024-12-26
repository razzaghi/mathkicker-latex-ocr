# Nougat-LaTeX-OCR

<img src="./asset/img2latex.jpeg" width="600">

Mathkicker-Nougat-LaTeX-based is fine-tuned from [facebook/nougat-base](https://huggingface.co/facebook/nougat-base) with [im2latex-100k] and a custom dataset to boost its proficiency in generating LaTeX code from images. 



## Uses
### fine-tune on your customized dataset
1. Prepare your dataset in [this](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO) format
2. Change ``config/base.yaml``
3. Run the training script
```python
python tools/train_experiment.py --config_file config/base.yaml --phase 'train'
```

### predict
1. [Download](https://huggingface.co/Norm/nougat-latex-base) the model
2. Install dependency
```bash
pip install -r all_requirements.txt
```
3. You can find an example in examples folder
```python
python examples/run_latex_ocr.py --img_path "examples/test_data/eq1.png"
```

### QA
- **Q:** Why did you copy and place the `image_processor_nougat.py` file in the repository rather than simply importing it from the `transformers` library if there are no changes compared to the one in `huggingface/transformers`?

- **A:** `transformers 4.34.0` is the first version that natively supports the nougat. However, there is a bug in the nougat processor within this version, which can result in a run failure. You can review the details of this issue [here](https://github.com/huggingface/transformers/issues/26597). Fortunately, the developers have already addressed this bug, and I anticipate that you will be able to directly import it from `transformers` in the next released version.

**please consider leaving me a star if you find this repo helpful :)**
