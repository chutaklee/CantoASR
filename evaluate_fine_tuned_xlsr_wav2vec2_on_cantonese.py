import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import argparse
import string

lang_id = "zh-HK" 
#model_id = "ctl/wav2vec2-large-xlsr-cantonese"
model_id = "./wav2vec2-large-xlsr-cantonese"
model_path = "./wav2vec2-large-xlsr-cantonese/checkpoint-30888"

chars_to_ignore_regex = '[\,\?\.\!\-\;\:"\“\%\‘\”\�\．\⋯\！\－\：\–\。\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']'
test_dataset = load_dataset("common_voice", f"{lang_id}", split="test") 
cer = load_metric("./cer")
processor = Wav2Vec2Processor.from_pretrained(f"{model_id}") 
model = Wav2Vec2ForCTC.from_pretrained(f"{model_path}") 
model.to("cuda")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

resamplers = {
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
}


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    sen = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    # convert 'D' and 'd' to '啲' if there a single 'D' or 'D' in the sentence
    # hacky stuff, wont work on 'D' or 'd' co-occure with normal english words
    # wont work on multiple 'D'
    
    if "d" in sen:
        if len([c for c in sen if c in string.ascii_lowercase]) == 1:
            sen = sen.replace("d", "啲")
    
    batch["sentence"] = sen
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resamplers[sampling_rate](speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=16)

print("CER: {:2f}".format(100 * cer.compute(predictions=result["pred_strings"], references=result["sentence"])))
