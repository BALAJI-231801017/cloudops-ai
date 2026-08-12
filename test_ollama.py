import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain high CPU usage in a simple way."
        }
    ]
)

print(response["message"]["content"])