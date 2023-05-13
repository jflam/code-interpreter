# code-interpreter

This repo contains a simple implementation of ChatGPT Code Interpreter from
OpenAI. It uses a local Jupyter kernel implementation to execute the code and
retrieve the results.

## Plan

- install CopilotX
- use IPython to communicate with a local Python kernel. First spike is to
  make sure that I can send code to the kernel and retrieve the results of the
  execution, or the errors. Build this using CopilotX and send feedback to the
  team based on my experiences.
- research and find an existing Chat front end that I can use to simulate the
  ChatGPT user interface
- build out the prompts that I can use to broker communications. ensure that
  I have the auto-fix loop implemented for errors that happen during exec
- build some scaffolding for evaluating the prompt effectiveness (perhaps)
  using tooling like humanloop or statsig