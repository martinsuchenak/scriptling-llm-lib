import sys
import llm

args = sys.argv[1:]
if len(args) < 1:
    print("Usage: sllm chat.py <model.gguf> [max_tokens] [strategy]")
    print("  strategy: greedy, temperature, top_k, top_p")
    print("  Type 'quit' to exit, 'clear' to reset session")
    sys.exit(1)

model_path = args[0]
max_tokens = int(args[1]) if len(args) > 1 else 50
strategy = args[2] if len(args) > 2 else "greedy"

session_id = "chat"
print("Chat session: " + session_id + " (model: " + model_path + ")")
print("Type 'quit' to exit, 'clear' to reset session")
print()

while True:
    line = sys.stdin.readline()
    if line == "":
        break
    line = line.rstrip("\n")
    if line == "quit":
        break
    if line == "clear":
        llm.clear_session(model_path, session_id)
        print("[session cleared]")
        print()
        continue
    if line == "":
        continue

    result = llm.generate(model_path, line, max_tokens, strategy,
        session=session_id, stats=True)
    print(result)
    print()

llm.clear_session(model_path, session_id)
