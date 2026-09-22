import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('scratch/markdown_matches.json', 'r', encoding='utf-8') as fp:
    data = json.load(fp)

keys = sorted(data.keys())
for f in keys[12:]:
    print(f"=== {f} ({len(data[f])}) ===")
    for m in data[f]:
        print(f"  L{m['line']} [{m['pattern']}] {m['text'][:110]}")
