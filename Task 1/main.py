import os
from pathlib import Path
from groq import Groq
import json

# importing the text file containing prompts
INPUT_FILE = Path(__file__).resolve().parent / "prompts.txt"
# declaring the .json file where the responses will go
OUTPUT_FILE = Path(__file__).resolve().parent / "responses.json"

# setting model that groq would allow us to use, did not allow a llama model so had to switch to openai
MODEL_NAME = "openai/gpt-oss-20b"

# function that uses file i/o to strip every line from the text file we feed it and return the list of the lines
def read_prompts(file_path):
    with file_path.open("r", encoding="utf-8") as input_file:
        prompts = []

        for line in input_file:
            prompt_text = line.strip()

            if prompt_text:
                prompts.append(prompt_text)
    return prompts

# now we connect to groq

# GROQ_API_KEY has been defined as an environment variable beforehand
api_key = os.getenv("GROQ_API_KEY")
# small check-case in case key was pasted wrong into the terminal, not important now
if not api_key:
    raise SystemExit("Set GROQ_API_KEY in your terminal before running.")

client = Groq(api_key=api_key)


# reading the prompts from the text file
prompts = read_prompts(INPUT_FILE)
if not prompts:
    raise SystemExit("No prompts were found in prompts.txt")

results = []

# using a for loop to print and store each question and its answer

for prompt_no,prompt_txt in enumerate(prompts,start=1):
    # printing the question
    print(f"\nSending question {prompt_no} of {len(prompts)}...")
    print(f"Question : {prompt_txt}")

    # fetching the answer
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt_txt,
            }
        ],
    )
    response_txt = chat_completion.choices[0].message.content
    results.append(
        {
            "prompt": prompt_txt,
            "response": response_txt,
        }
    )
    print(f"Answer: {response_txt}")
print(f"\nFinished! Collected {len(results)} responses.")

# saving the responses in responses.json
with OUTPUT_FILE.open("w",encoding="utf-8") as output_file:
    json.dump(results,output_file,indent=2,ensure_ascii=False)