from dotenv import load_dotenv

import os
import logging
from datetime import datetime
from litellm import completion

load_dotenv("../src/.env")

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f"{os.path.splitext(os.path.basename(__file__))[0]}_{datetime_str}.log",
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    filemode='w',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)


def prompt_model(user_prompt, system_prompt=None, model='gpt-4o-2024-11-20'):

  chat_completion = litellm.completion(
      messages=[
          {
              "role": "user",
              "content": prompt,
          },
            {
                "role": "system",
                "content": system_prompt,
            }
      ],
      model=model,
  )
  a =  chat_completion.choices[0].dict()['message']['content']
  return a


user_prompt = f"""
INSTRUCTIONS
Generate a list of terms related to AI, specifically user-facing use cases of large language models.

EXAMPLES
- artificial intelligence
- AI
- chatbot
- large language model 
- llm 

"""
