def doctor_inference(image, prompt=None):
    if prompt is not None:
        question = prompt
    else:
        question = 'Analyze the provided medical image. Determine the imaging modality (e.g., MRI, X-ray, CT scan, ultrasound) and identify the organ or body part in the image. Conduct a detailed examination of the image to detect any abnormalities, including but not limited to fractures, tumors, infections, or degenerative changes. Provide an analysis and description of the findings, and suggest potential treatments or next steps if abnormalities are present.'
    msgs = [{'role': 'user', 'content': [image, question]}]

    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
           res = model.chat( image=image, msgs=msgs, tokenizer=tokenizer, sampling=True, temperature=0.95, stream=True )

    generated_text = ""
    generated_text = 'foo'
    for new_text in res: generated_text += new_text 
    return generated_text


def checkup(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = get_image(request)  
        prompt = request.POST.get('text') or None

        if image is not None: 
            if prompt is None:
                prescription = doctor_inference(image)
            else:
                prescription = doctor_inference(image, prompt)

        return JsonResponse({
            'prescrition': 'foo'
        })

    return render(request, 'captioner/doctor.html')




def load_doctor(model_name):
    model = AutoModel.from_pretrained( "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", trust_remote_code=True)    return model, tokenizer