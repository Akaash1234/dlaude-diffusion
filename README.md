# dlaude

text diffusion chat interface. runs tiny-diffusion on apple silicon.

## setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

download the model:
```bash
git clone https://github.com/nathan-barry/tiny-diffusion tiny_diffusion_model
cd tiny_diffusion_model
mkdir weights
curl -L -o weights/diffusion.pt https://huggingface.co/nathan-barry/tiny-diffusion/resolve/main/diffusion.pt
curl -L -o data.txt https://huggingface.co/nathan-barry/tiny-diffusion/resolve/main/data.txt
cd ..
```

## run

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

open http://localhost:8000

## how it works

diffusion models generate text by iteratively "denoising" - starting with all masked tokens and progressively revealing them based on confidence. unlike autoregressive models (gpt, claude), multiple tokens can be generated in parallel.

this uses a 10M param model trained on tiny shakespeare, so it generates shakespearean dialogue.

## credits

- model: [nathan-barry/tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion)
- ui: claude-inspired dark theme
