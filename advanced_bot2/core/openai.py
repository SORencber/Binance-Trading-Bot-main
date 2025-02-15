
from openai import OpenAI
from config.bot_config import OPENAI_API_KEY


async def  openai_connect(prompt,openai_model):
                
    client = OpenAI(api_key=OPENAI_API_KEY)
    response_text = ""
    stream = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content":prompt}],
                    stream=True,
                )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            response_text += chunk.choices[0].delta.content
               
               
    if not response_text:
        response_text = "⚠ CinCon yanıt veremedi."
        return response_text
    else:
        return response_text
