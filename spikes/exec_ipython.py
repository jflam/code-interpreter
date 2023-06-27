SAMPLE_CODE="""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Sine Function")
plt.xlabel("x")
plt.ylabel("sin(x)")
"""

from jupyter_client import KernelManager
from queue import Empty
import base64

# Start a new kernel
km = KernelManager()
km.start_kernel()

# Get a client to interact with the kernel
kc = km.client(block=True)
kc.execute(SAMPLE_CODE)

# Fetch the output
output = None
while True:  # continue fetching messages until 'execute_reply' is received
    try:
        output = kc.get_iopub_msg(timeout=10)  # increase timeout if necessary
    except Empty:
        break

    msg_type = output['msg_type']

    if msg_type == 'execute_reply':
        print(output)
        break
    elif msg_type == 'display_data':
        image_data = output['content']['data']['image/png']
        with open('hello_world.png', 'wb') as f:
            f.write(base64.b64decode(image_data))

# Shut down the kernel
km.shutdown_kernel(now=True)
kc.stop_channels()
del kc
del km