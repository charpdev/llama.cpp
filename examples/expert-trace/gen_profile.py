#!/usr/bin/env python3
"""
Run expert-trace on a set of prompts and produce a hot-expert profile.

Design notes (from tan_llama/notes/):
- top-k defaults to n_expert_used (8 for Qwen3.5-122B) not 20%, because that
  matches the actual routing budget per token and gives the tightest hot set.
- Prompts cover multiple families (coding, math, knowledge, reasoning) so the
  profile captures routing diversity, not just one workload type.
- -n 64 generates real decode tokens; decode-phase routing differs from prefill
  and dominates wall-clock time in interactive use.
- The profile format is consumed by --hot-expert-profile in llama-server.

Usage:
  python3 gen_profile.py \
    --model /path/to/model.gguf \
    --model-id qwen35-122b-q4km \
    --trace-bin /path/to/llama-expert-trace \
    [--top-k 8] [--n-experts 256] [--n-prompts 20]
"""
import argparse, collections, os, re, subprocess

# Benchmark-inspired prompt set covering multiple routing families.
# Coding/repair prompts tend to activate a different expert subset from
# math/knowledge/reasoning - keeping all families gives a broader hot set.
PROMPTS = [
    # coding
    "Write a Python function to sort a list of integers.",
    "Write a binary search implementation in C++.",
    "Implement a linked list in Python.",
    "Write a function to reverse a string in place.",
    "Write a recursive fibonacci function.",
    "Write a merge sort implementation.",
    "Write a function to detect a cycle in a linked list.",
    "Write a binary tree traversal in Python.",
    "Write a REST API endpoint in Python Flask.",
    "Write a SQL query to find duplicate rows in a table.",
    # knowledge / explanation
    "Explain how transformers work in deep learning.",
    "What is the difference between TCP and UDP?",
    "Explain the CAP theorem.",
    "What is a hash table and how does it handle collisions?",
    "Explain Docker containers vs virtual machines.",
    "What is the difference between process and thread?",
    # math / reasoning
    "How does gradient descent work?",
    "Explain backpropagation.",
    "What is a deadlock and how do you prevent it?",
    "Write a regex to validate an email address.",
]

def parse_trace(output: str) -> dict:
    """Parse expert_trace stdout -> {layer: Counter(expert_id)}"""
    counts = collections.defaultdict(collections.Counter)
    for line in output.splitlines():
        m = re.match(r"expert_trace layer=(\d+)", line)
        if not m:
            continue
        layer = int(m.group(1))
        for experts in re.findall(r"\[([0-9,]+)\]", line):
            for eid in experts.split(","):
                counts[layer][int(eid)] += 1
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True, help="e.g. qwen35-122b-q4km")
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--top-k", type=int, default=None,
                    help="hot experts per layer (default: n_expert_used, i.e. 8 for Qwen3.5-122B)")
    ap.add_argument("--n-experts", type=int, default=256,
                    help="total experts in model (default: 256)")
    ap.add_argument("--n-expert-used", type=int, default=8,
                    help="experts routed per token (default: 8)")
    ap.add_argument("--n-tokens", type=int, default=64,
                    help="decode tokens per prompt (default: 64)")
    ap.add_argument("--trace-bin", required=True,
                    help="path to llama-expert-trace binary")
    args = ap.parse_args()

    top_k = args.top_k or args.n_expert_used  # tight hot set by default

    out_dir = os.path.join(os.path.dirname(__file__), "profiles", args.model_id)
    os.makedirs(out_dir, exist_ok=True)

    prompts = PROMPTS[:args.n_prompts]
    all_counts = collections.defaultdict(collections.Counter)

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] tracing: {prompt[:60]}...")
        cmd = [
            args.trace_bin,
            "-m", args.model,
            "--cpu-moe",
            "-ngl", "99",
            "-n", str(args.n_tokens),
            "-p", prompt,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        for layer, ctr in parse_trace(result.stdout).items():
            all_counts[layer].update(ctr)

    profile_path = os.path.join(out_dir, "hot_profile.txt")
    with open(profile_path, "w") as f:
        f.write(f"# Model: {args.model_id}\n")
        f.write(f"# Prompts: {len(prompts)}, decode_tokens_per_prompt: {args.n_tokens}\n")
        f.write(f"# top_k: {top_k} of {args.n_experts} experts per layer\n")
        f.write(f"# Format: <layer> <expert_id> ...\n")
        for layer in sorted(all_counts.keys()):
            top = [eid for eid, _ in all_counts[layer].most_common(top_k)]
            f.write(f"{layer} " + " ".join(str(e) for e in sorted(top)) + "\n")

    print(f"\nProfile saved: {profile_path}")
    print(f"Layers: {len(all_counts)}, top_k per layer: {top_k}")

if __name__ == "__main__":
    main()
