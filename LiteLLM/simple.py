from litellm import completion
import os

models = [
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4-5",
]

for model in models:
    print(f"\n# {model}\n")
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a marketing expert."},
            {
                "role": "user",
                "content": "What would be a good company name for a company that makes colorful socks?",
            },
        ],
    )
    print(response.choices[0].message.content)
