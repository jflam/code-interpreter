import cexprtk, logging, openai, os, discord
from discord import Client, Intents, app_commands
from jupyter_client import KernelManager, BlockingKernelClient
from queue import Empty
import base64

ASKAI_BOT_TOKEN=os.environ['ASKAI_BOT_TOKEN']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
MODEL_NAME="gpt-3.5-turbo"
BOT_NAME="Jarvis"
TEMPERATURE=0.5
SYSTEM="""
You are a helpful assistant to a user.
"""
SAMPLE_CODE="""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Sine Function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
"""

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = Intents.default()
intents.message_content = True

client = Client(intents=intents)
tree = app_commands.CommandTree(client)

# Initialize Jupyter kernel for code execution
km = KernelManager()
km.start_kernel()
kc = km.client(block=True)

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

def execute_code(kc: BlockingKernelClient, code: str):
    kc.execute(code)

    execute_result = None
    display_data = None 

    # The result of the execution is 0 or more messages on the 'iopub' channel
    # followed by a status message that says the kernel is now idle (I
    # believe). We need to wait for the status message and accumulate all the
    # 'display_data' and 'execute_result' messages that come before it.
    message = None
    while True: 
        try:
            message = kc.get_iopub_msg(timeout=10)  
        except Empty:
            return None

        msg_type = message['msg_type']
        if msg_type == 'execute_result':
            execute_result = message
        elif msg_type == 'display_data':
            display_data = message
        elif msg_type == 'status':
            execution_state = message['content']['execution_state']
            if execution_state == 'idle':
                break
            elif execution_state == 'busy':
                continue
            else:
                print("WARNING: Unexpected status message")
                print(message)

    if display_data is not None:
        data_payload = display_data['content']['data']
        image_data = data_payload.get('image/png')
        if image_data is not None:
            with open('hello_world.png', 'wb') as f:
                f.write(base64.b64decode(image_data))
            return ('image/png', 'hello_world.png', display_data)
    elif execute_result is not None:
        data_payload = execute_result['content']['data']
        text_data = data_payload.get('text/plain')
        if text_data is not None:
            return ('text/plain', text_data, execute_result)
    else:
        raise Exception("No result or display data")

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

    # Hard-coded execution of Python code that generates a matplotlib plot
    # using the Jupyter Python kernel
    mime_type, result, msg = execute_code(kc, SAMPLE_CODE)
    current_dir = os.getcwd()
    if mime_type == 'image/png':
        file = discord.File(f"{current_dir}/{result}", filename=result)
        embed = discord.Embed()
        embed.set_image(url=f"attachment://{result}")
        await message.channel.send(embed=embed, file=file)
    elif mime_type == 'text/plain':
        await message.channel.send(result)
    else:
        await message.channel.send("Unknown result from Jupyter kernel")

    result = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )
    response = result.choices[0].message.content
    await message.channel.send(response)

client.run(ASKAI_BOT_TOKEN)