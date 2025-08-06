import os

api_key = os.environ.get("OPENAI_API_KEY")

if api_key:
    print(f"Success! The OPENAI_API_KEY is set: {api_key}")
else:
    print("Error: The OPENAI_API_KEY is not set.")
