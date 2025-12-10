# pip install openai>=1.0
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
)


def build_prompt(selected_classes):
    classes_str = ", ".join(selected_classes)
    
    prompt = f'''
There are some categories: {classes_str}. Please write prompts that contain these objects
for the image generation task. Please provide me with 10 examples.
Here are some requirements:
(1) The background description of the image can be more complex and not too monotonous.
(2) The scenarios and details for these examples are as diverse as possible.
(3) There can be more detailed descriptions and attributes for the category such as color,
posture, and actions.
(4) The details of different examples should be as diverse as possible.

Please provide the 10 examples in the following format:
1. [prompt description]
2. [prompt description]
...
10. [prompt description]
'''
    return prompt

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"errorr: {e}")
        return None

def parse_generated_texts(text, selected_classes):
    results = []
    lines = text.split('\n')
    
    for line in lines:
        if line.strip() and line[0].isdigit() and '. ' in line:
            
            parts = line.split('. ', 1)
            if len(parts) == 2:
                description = parts[1].strip()
                class_names = ",".join(selected_classes)
                results.append(f"{class_names}: {description}")
    
    return results

def main():
    total_iterations = 10000
    magicseg_file = "magicseg.txt"
    classes = load_magicseg_classes(magicseg_file)
    
    all_results = []
    for _ in range(total_iterations):
        selected_classes = random_select_classes(classes)
        
        prompt = build_prompt(selected_classes)

        
        text = generate_text(prompt)
        if text:
            
            parsed_results = parse_generated_texts(text, selected_classes)
            all_results.extend(parsed_results)
            
        else:
            print("error")
    
    output_file = "generated_texts.txt"
    with open(output_file, 'a+', encoding='utf-8') as f:
        for result in all_results:
            f.write(result + "\n")
