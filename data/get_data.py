import json
import openai
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class Item(BaseModel):
    question: str = None
    chat_history: list = None


with open("GPT_SECRET_KEY.json") as f:
    secret_key = json.load(f)

openai.api_key = secret_key["API_KEY"]


def gpt3(prompt):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    print(response)
    answer = response.choices[0]["text"]
    return answer


@app.post('/gpt')
def get_answer(request_data: Item):
    prompt = ''  # 'I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".\n\n'
    cnt = 0
    for sentence in request_data.chat_history:
        if cnt % 2 == 0:
            prompt += 'Q: ' + sentence + '\n'
        else:
            prompt += 'A: ' + sentence + '\n\n'
        cnt += 1
    prompt += 'Q: ' + request_data.question + '\nA:'
    answer = gpt3(prompt)
    res = {'answer': answer}
    return res
