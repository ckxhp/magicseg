import os
import random
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import re



def create_filename(selected_classes):
    return "_".join(selected_classes)

def replace_classes_with_nothing(text, selected_classes):
    for class_name in selected_classes:
        text = re.sub(r'\b' + re.escape(class_name) + r'\b', 'nothing', text, flags=re.IGNORECASE)
    return text

def load_sd_pipeline():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        
        return pipe
    except Exception as e:
        print(f"error: {e}")
        return None

def generate_image(pipe, prompt, filename, output_dir="generated_images"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
        
        image_path = os.path.join(output_dir, f"{filename}.jpg")
        image.save(image_path)
        
        return image_path
    except Exception as e:
        return None

def process_generated_texts(input_file="generated_texts.txt"):
    if not os.path.exists(input_file):
        print(f"error {input_file}")
        return
    
    pipe = load_sd_pipeline()
    if pipe is None:
        return
    
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for i, line in enumerate(lines):

        if ': ' not in line:
            continue
        
        class_names_part, description = line.split(': ', 1)
        selected_classes = class_names_part.split(',')
        
        filename = create_filename(selected_classes)
        
        original_path = generate_image(pipe, description, filename, "original_images")
        
        if original_path:
            neg_description = replace_classes_with_nothing(description, selected_classes)
            neg_filename = f"{filename}-neg"

            
            generate_image(pipe, neg_description, neg_filename, "neg_images")


def main():

    if not os.path.exists("generated_texts.txt"):
        return
    
    process_generated_texts()


if __name__ == "__main__":
    main()