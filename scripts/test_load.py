import torch
from transformers import T5ForConditionalGeneration
try:
    model = T5ForConditionalGeneration.from_pretrained("models/tiny-tab-v1/checkpoint-93780")
    print("Success loading model")
except Exception as e:
    import traceback
    traceback.print_exc()
