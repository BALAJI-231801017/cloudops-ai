from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Explain high CPU usage in a simple way."
)

print(response.output_text)