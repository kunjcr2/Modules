import openai
import speech_recognition as sr

class ChatGPT:
    def __init__(self, api_key, model="gpt-4-mini"):
        openai.api_key = api_key
        self.model = model

    def chat(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
