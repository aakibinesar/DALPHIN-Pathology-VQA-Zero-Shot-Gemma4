# Builds body_imgfirst.py from body.py (body.py itself is not touched): the ONLY change is that the images come BEFORE the question text in every prompt (Gemma-4 model card best practice; all earlier Gemma-4 runs had the text first).
# Everything else (greedy decoding, instruction suffix, post-processing, token budget 560, no thinking) stays as in the t560-a run, so the result is directly comparable with it.
src = open("body.py", encoding="utf-8").read()
a = '    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)\n'
assert src.count(a) == 1
src = src.replace(a, '    content = [c for c in content if c["type"] == "image"] + [c for c in content if c["type"] != "image"]      # images before the text\n' + a)
open("body_imgfirst.py", "w", encoding="utf-8").write(src); print("body_imgfirst.py written", len(src), "chars")
