I need you to wire a trained PyTorch model into this website's FastAPI backend so that an uploaded microscope image is classified as ANA-Positive or ANA-Negative with a confidence. I have a demo tomorrow, so the priority is a correct, working end-to-end flow, not extra features.

## Background

This is a final-year project on ANA (antinuclear antibody) testing. A lab photographs HEp-2 cells under a fluorescence microscope; a positive sample glows green in the cells, a negative one is almost dark. I trained a binary image classifier for this in a separate project. The website flow is: a user uploads one image, the backend runs the model, the frontend shows the label and the confidence. The uploaded images are new to the model and can be any common format and any size.

The model was trained with one exact preprocessing pipeline. If the backend preprocesses differently, the model still returns an answer but it is silently wrong, so reproducing the pipeline below exactly is the most important part of this task.

## What I am giving you

One weights file, `best.pt` (about 16 MB). It is already saved in the `results` folder of this project; find it there and load it from that location rather than moving it. If there is more than one `results` folder or more than one `.pt` file and it is not obvious which is meant, ask me. Facts about it:

- Architecture: EfficientNet-B0 from the `timm` library, with a single output (`num_classes=1`).
- The file is a `state_dict` only (saved with `torch.save(model.state_dict(), ...)`), not a pickled model. The architecture must be built first and the weights loaded into it.
- The single output is a logit. Positive is class 1.
- Temperature for calibrated confidence: `1.754` (probability = sigmoid(logit / 1.754)).
- Decision cut-off: probability >= 0.5 means ANA-Positive.

## The exact pipeline (reference implementation)

Use this logic as it is. You may restructure it to fit the codebase, but do not change any step, constant, or the order.

```python
import io
import math

import numpy as np
import timm
import torch
from PIL import Image

IMG_SIZE = 512
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TEMPERATURE = 1.754
THRESHOLD = 0.5


def load_model(weights_path, device="cpu"):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)          # strict: every key must match
    return model.to(device).eval()


def preprocess(image_bytes):
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    scale = IMG_SIZE / min(w, h)                                   # shorter side -> 512, aspect ratio kept
    im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    w, h = im.size
    left = int(round((w - IMG_SIZE) / 2.0))                        # centre 512x512 crop
    top = int(round((h - IMG_SIZE) / 2.0))
    im = im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)   # HWC -> CHW, 0..1
    return ((x - MEAN) / STD).unsqueeze(0)                         # shape (1, 3, 512, 512)


@torch.inference_mode()
def predict(model, image_bytes, device="cpu"):
    logit = model(preprocess(image_bytes).to(device)).item()
    p_positive = 1.0 / (1.0 + math.exp(-logit / TEMPERATURE))
    is_positive = p_positive >= THRESHOLD
    return {
        "label": "ANA-Positive" if is_positive else "ANA-Negative",
        "confidence": p_positive if is_positive else 1.0 - p_positive,   # 0.5 .. 1.0
        "probability_positive": p_positive,
    }
```

Rules that matter, and why:

- No contrast stretching, histogram equalisation, auto-levels, sharpening or brightness adjustment anywhere. Brightness is the signal the model uses; a dark image is supposed to stay dark.
- Channel order is RGB. If any existing code reads images with OpenCV, that is BGR and must not be fed to the model.
- `pretrained=False` when creating the model. The weights come from my file; nothing should be downloaded at startup.
- `model.eval()` is required. Without it, batch-norm and dropout behave as in training and the outputs are wrong and non-deterministic.
- `map_location` so that it loads on a machine without a GPU. CPU inference is fine for the demo; use CUDA only if it is available.
- Images whose shorter side is under 512 are upscaled by the same resize step. That is expected.
- The model sees the centre square of the image. That is how it was evaluated.

## What to do

1. Read the backend first. Find where a model or `.pt` file is currently loaded or stubbed, which endpoint the frontend calls for classification, and what response shape the frontend expects. Keep that contract: map my `label` / `confidence` / `probability_positive` onto whatever field names the frontend already reads, rather than changing the frontend. If the expected shape is unclear or conflicts with these three values, ask me.
2. Put the inference code in one module and load the model once at application startup (FastAPI lifespan or equivalent), not on every request. Take the weights path from configuration or an environment variable, defaulting to `best.pt` in the `results` folder, and resolve it relative to the project rather than the current working directory so that it works wherever the server is started from.
3. In the endpoint, read the uploaded file's bytes and call the pipeline. PyTorch inference is blocking, so use a plain `def` endpoint or run it in a threadpool so that it does not block the event loop.
4. Handle bad input cleanly: a file that is not a readable image should return a 4xx with a clear message, not a 500. Accept at least PNG, JPG/JPEG, TIF/TIFF and BMP. If a 16-bit image arrives (PIL modes `I;16`, `I`, `F`), scale it to 8-bit before `convert("RGB")`, because PIL clips those modes instead of rescaling.
5. Add what is missing to the backend's dependency file: `torch`, `timm` (1.0 or newer), `pillow`, `numpy`, and `python-multipart` if uploads are not already working. Do not add `torchvision`; the pipeline above does not need it.
6. Confirm how confidence is displayed. The value is a fraction between 0.5 and 1.0; if the frontend shows a percentage, convert it once and in one place.

## How to verify before telling me it is done

- Start the backend and confirm the model loads with no missing or unexpected keys and no network download.
- Send the same image twice and confirm the result is identical.
- Call the endpoint through the real upload path (not only the Python function) and show me the JSON response.
- I will give you a few test images with their known labels. Run them and report label and confidence for each. If you have none yet, ask me for them.
- Tell me plainly about anything you could not test.

## Constraints

- Do not retrain, fine-tune, or modify the model, and do not change the preprocessing to "improve" results.
- My test images come from a research dataset whose licence forbids redistribution. Do not commit them to the repository or upload them anywhere; keep them out of git.
- The result is for research and demonstration only, not for diagnosis. If the UI has a natural place for a one-line note saying so, add it; do not redesign anything for it.
- Keep changes focused on making this flow work. Do not refactor unrelated code.

When you finish, give me a short summary: which files you changed, the exact path the weights are loaded from, the command to run the backend, and the results of the checks above.
