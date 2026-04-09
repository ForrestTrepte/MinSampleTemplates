import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1000,
    system="You are a marketing expert.",
    messages=[
        {
            "role": "user",
            "content": "What would be a good company name for a company that makes colorful socks?",
        }
    ],
)
print(message.content[0].text)
