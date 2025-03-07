from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="o3-mini",
    store=True,
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)
print(completion.choices[0].message.content)
