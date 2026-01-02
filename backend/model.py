import random
import time

class Dlaude:
    def __init__(self):
        self.responses = [
            "that's interesting, tell me more",
            "i see what you mean",
            "could you elaborate?",
            "based on my training data...",
            "i'm just a tiny diffusion model",
            "diffusion is cool",
            "generating noise...",
            "denoising steps: 100%"
        ]

    def generate(self, prompt, history):
        time.sleep(1.5)
        return random.choice(self.responses)
