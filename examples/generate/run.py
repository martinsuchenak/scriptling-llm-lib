import sys
import llm

args = sys.argv[1:]
if len(args) < 2:
    print("Usage: sllm run.py <model.gguf> \"prompt\" [max_tokens] [strategy] [stats]")
    print("  strategy: greedy, temperature, top_k, top_p")
    sys.exit(1)

show_stats = False
template_name = ""
repeat_penalty = 1.15
filtered = []
i = 0
while i < len(args):
    a = args[i]
    if a == "stats":
        show_stats = True
    elif a == "-t" and i + 1 < len(args):
        template_name = args[i + 1]
        i = i + 1
    elif a == "-rp" and i + 1 < len(args):
        repeat_penalty = float(args[i + 1])
        i = i + 1
    else:
        filtered.append(a)
    i = i + 1

model_path = filtered[0]
prompt = filtered[1]
max_tokens = int(filtered[2]) if len(filtered) > 2 else 50
strategy = filtered[3] if len(filtered) > 3 else "greedy"

print("Loading " + model_path + "...")
print()

result = llm.generate(model_path, prompt, max_tokens, strategy,
    repeat_penalty=repeat_penalty, template=template_name,
    stats=show_stats)
print(result)
