# code-interpreter

This repo contains a simple implementation of ChatGPT Code Interpreter from
OpenAI. It uses a local Jupyter kernel implementation to execute the code and
retrieve the results. It is implemented as a Discord bot to avoid any need
to create a user interface.

## Plan

- DONE: install CopilotX
- DONE: use `jupyter_client` to communicate with a local Python kernel. First
  spike is to make sure that I can send code to the kernel and retrieve the
  results of the execution, or the errors. Build this using CopilotX and send
  feedback to the team based on my experiences.
- DONE: research and find an existing Chat front end that I can use to
  simulate the ChatGPT user interface -- using Discord
- DONE: create an `execute_code` function that uses Jupyter client to 
  execute code in the kernel and process both text-based and image-based 
  (plot) results.
- DONE: integrate `execute_code` with discord bot
- detect whether ChatGPT response contains code and add some UI in Discord 
  to let the user click to execute the code
- build out the prompts that I can use to broker communications. ensure that I
  have the auto-fix loop implemented for errors that happen during exec
- build some scaffolding for evaluating the prompt effectiveness (perhaps)
  using tooling like humanloop or statsig
- Inject arbitrary content (e.g., a cost counter) into each response
- classification of intent against a set of tools like code interpreter
- for code interpreter scenarios, generate code and execute locally using
  IPython kernel
- use chain of thought to synthesize and return results