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

from jupyter_client import KernelManager, BlockingKernelClient
from queue import Empty
import base64

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

# Start a new kernel
km = KernelManager()
km.start_kernel()
kc = km.client(block=True)

mime_type, result, message = execute_code(kc, SAMPLE_CODE)
if mime_type == "text/plain":
    print(f"RESULT: {result}")
elif mime_type == "image/png":
    print(f"IMAGE FILENAME: {result}")

# Shut down the kernel and clean up
km.shutdown_kernel(now=True)
kc.stop_channels()
del kc 
del km