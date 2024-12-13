import json
import openai
from openai import OpenAI
import re

import os
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()
DEFAULT_PROMPT_INPUT_TAG = '[input]'

calculator_prompt = f"""
Your task is to add calls to a Calculator API to a piece of text.
The calls should help you get information required to complete the text. 
You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. 
Here are some examples of API calls:
Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.
Input: The population is 658,893 people. This is 11.4% of the national average of 5,763,868 people.
Output: The population is 658,893 people. This is 11.4% of the national average of [Calculator(658,893 / 11.4%)] 5,763,868 people.
Input: A total of 252 qualifying matches were played, and 723 goals were scored (an average of 2.87 per match). This is three times less than the 2169 goals last year.
Output: A total of 252 qualifying matches were played, and 723 goals were scored (an average of [Calculator(723 / 252)] 2.87 per match). This is twenty goals more than the [Calculator(723 - 20)] 703 goals last year.
Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011 - 1994)] 17 years.
Input: From this, we have 4 * 30 minutes = 120 minutes.
Output: From this, we have 4 * 30 minutes = [Calculator(4 * 30)] 120 minutes.
Input: There are 22 walnut trees currently in the park. 
Park workers will plant walnut trees today. 
When the workers are finished there will be 55 walnut trees in the park. 
How many walnut trees did the workers plant today? 
The workers planted 33 walnut trees today.
Output: There are 22 walnut trees currently in the park. 
Park workers will plant walnut trees today. 
When the workers are finished there will be 55 walnut trees in the park. 
How many walnut trees did the workers plant today? 
The workers planted [Calculator(55 - 22)] 33 walnut trees today.
Input: Zoe had 6 songs on her mp3 player. 
She deleted 3 old songs from it. 
That means she now has 3 songs on her mp3 player. 
She added 20 new songs to it. 
That means she now has 23 songs on her mp3 player.
Output: Zoe had 6 songs on her mp3 player. 
She deleted 3 old songs from it. 
That means she now has [Calculator(6 - 3)] 3 songs on her mp3 player. 
She added 20 new songs to it. 
That means she now has [Calculator(3 + 20)] 23 songs on her mp3 player.
Input: In the first round, she scored 16 points. So we start with 16. 
In the second round, she scored 33 points. So we add 33 to 16 and get 49. 
In the last round, she lost 48 points. So we subtract 48 from 49 and get 1. 
Therefore, she had 1 point at the end of the game.
Output: In the first round, she scored 16 points. So we start with 16. 
In the second round, she scored 33 points. So we add 33 to 16 and get [Calculator(33 + 16)] 49. 
In the last round, she lost 48 points. So we subtract 48 from 49 and get [Calculator(49 - 48)] 1. 
Therefore, she had 1 point at the end of the game.
Input: {DEFAULT_PROMPT_INPUT_TAG}
Output:
"""

def gpt_model_call(model = 'gpt-4o-mini', prompt = ''):
    #print(prompt)
    completion = client.chat.completions.create(
    model=model,
    messages=[
            {"role": "system", "content": "You are a helpful assistant. Follow the provided examples as instructions and demonstrations. Provide your answer directly, without restating the question or input."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    content = completion.choices[0].message.content
    print(content)
    print("len prompt:", len(prompt))
    print("len content:", len(content))
    return content

# Load the JSON file
file_path = "./teacher_completion_data/B_text-davinci-002__C_zs_cot/D_multiarith.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Access metadata
metadata = data['metadata']
print("Metadata:", metadata)
copied_data = data.copy()
# Access the question and answer in data
for i, (key, samples) in enumerate(data['data'].items()):
    for j, sample in enumerate(samples):
        question = sample['question']
        answer = sample['answer']
        reasoning_completion = sample['reasoning_completion']
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Reasoning_completion: {reasoning_completion}")
        try:
            prompt = re.sub(re.escape(DEFAULT_PROMPT_INPUT_TAG), reasoning_completion, calculator_prompt)
            print("#"*80)
            refined_completion = gpt_model_call(prompt=prompt)
            print("#"*80)
            copied_sample = sample.copy()
            copied_sample['reasoning_completion'] = refined_completion
            copied_sample['prompt'] = copied_sample['prompt'].replace(reasoning_completion, refined_completion)
            copied_data['data'][key][j] = copied_sample
        except Exception as e:  # Catch any exception
            print(f"An error occurred: {e}")
    # if i >= 10:
    #     break



file_path_copied = "./teacher_completion_data/B_text-davinci-002__C_zs_cot/D_multiarith_copied.json"
with open(file_path_copied, 'w') as file:
    json.dump(copied_data, file, indent=4)

print(f"JSON data has been saved to {file_path_copied}")