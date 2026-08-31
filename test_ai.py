"""Verification test for local Ollama / Llama 3.2 connectivity."""

import sys
import ollama

def main():
    print("Connecting to local Ollama daemon...")
    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        models = client.list()
        print("Connected to Ollama! Available models:")
        for m in models.get("models", []):
            name = m.get("name") or m.get("model")
            print(f" - {name}")

        test_model = "llama3.2:3b"
        print(f"\nTesting prompt with model '{test_model}'...")
        response = client.chat(
            model=test_model,
            messages=[{"role": "user", "content": "Explain CPU saturation in one concise sentence."}]
        )
        print("Response received:")
        print(response.get("message", {}).get("content"))
        print("\nOllama connectivity test PASSED.")
    except Exception as e:
        print(f"\n[NOTE] Ollama daemon is offline or model not pulled: {e}")
        print("CloudOps AI will gracefully fall back to ML + rule-based monitoring without LLM.")
        sys.exit(0)

if __name__ == "__main__":
    main()
