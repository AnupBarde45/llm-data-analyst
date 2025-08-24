import requests

def test_ollama():
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": "Convert this to SQL: Show all products",
            "stream": False
        })
        print("✅ Ollama is running!")
        print("Response:", response.json()["response"][:100])
        return True
    except Exception as e:
        print("❌ Ollama connection failed:", str(e))
        return False

if __name__ == "__main__":
    test_ollama()