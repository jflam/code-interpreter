import cexprtk, logging, openai, os
from discord import Client, Intents, app_commands

ASKAI_BOT_TOKEN=os.environ['ASKAI_BOT_TOKEN']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
MODEL_NAME="gpt-3.5-turbo"
BOT_NAME="Jarvis"
TEMPERATURE=0.5
SYSTEM="""
You are a helpful assistant to a user.
"""

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = Intents.default()
intents.message_content = True

client = Client(intents=intents)
tree = app_commands.CommandTree(client)

@client.event 
async def on_ready():
    results = await tree.sync()
    print(f"logged in as {client.user}\nSync results:\n{results}")

@tree.command(name="calculate", description="Calculate an expression")
@app_commands.checks.bot_has_permissions(send_messages=True)
async def calculate(interaction, expression: str):
    try:
        result = cexprtk.evaluate_expression(expression, {})
        await interaction.response.send_message(f"Result: {result}")
    except Exception as e:
        await interaction.response.send_message(f"Error: {str(e)}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    history = [message async for message in message.channel.history(limit=10)]
    history = [x for x in history if x is not None]
    history.reverse()

    messages = [
        { 
            "role": "system",
            "content": SYSTEM
        }
    ]
    for message in history:
        author = message.author.name
        messages.append(
            {
                "role": "user" if author == BOT_NAME else "assistant",
                "content": message.content
            }
        )

    # Compute the number of tokens and output that
    await message.channel.send("*Thinking*...")

    result = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )
    response = result.choices[0].message.content
    await message.channel.send(response)

client.run(ASKAI_BOT_TOKEN)