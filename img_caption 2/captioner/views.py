import os
import gc
import time
from PIL import Image
from tqdm.auto import tqdm
from gtts import gTTS

import torch 
from huggingface_hub import login
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 


from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage


torch.cuda.empty_cache()
torch.cuda.empty_cache()
gc.collect()
gc.collect()

login("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")

model_name = 'microsoft/Phi-3.5-vision-instruct'
doctor_name = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1"


def load_phi3(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation='flash_attention_2').eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True,num_crops=4)
    return model, processor


model, processor = load_phi3(model_name=model_name)



def get_image(request):
        if request.method == 'POST' and request.FILES['image']:
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_path = fs.url(filename)
            image = Image.open(os.path.join(fs.location, filename))
            return image

 
        return render(request, 'captioner/upload.html')



def inference(image):
        images = []
        images.append(image)
        placeholder = f"<|image_{1}|>\n"
        messages = [
        {"role": "user", "content": placeholder+"Descirbe the image to a blind person in way that he gets to know all the objects present around him"},
        ]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 
        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        }   

        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        del generate_ids
        torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

        return response


def gtts_inference(text):
    output_path = os.path.join(settings.MEDIA_ROOT, 'tts_audio', 'speech.mp3')  # Save as mp3
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    tts = gTTS(text=text, lang='en', slow=False)


    tts.save(output_path)

    gc.collect()
    gc.collect()

    return output_path


def to_audio(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = get_image(request)  

        if image is not None:
            text = inference(image)
            
            audio_path = gtts_inference(text)

        
            relative_audio_path = os.path.relpath(audio_path, settings.MEDIA_ROOT)
            audio_url = settings.MEDIA_URL + relative_audio_path

        
            return JsonResponse({
                'audio_url': audio_url
            })

    return render(request, 'captioner/upload.html')
