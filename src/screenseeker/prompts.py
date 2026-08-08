"""The two ScreenSeekeR prompts, adapted from the paper (Appendix C, Tables 7-8)."""

POSITION_INFERENCE = """\
I want to identify a UI element that best matches my instruction. Help me decide
which region(s) of the screenshot to focus on, and list UI elements that might
appear next to the target.

Output requirements:
1. List possible regions in DESCENDING order of probability.
2. Make specific, unique references. "Other icons" or "window" are NOT allowed.
3. For each region, give an approximate bounding box in PIXELS of THIS image as
   [x1, y1, x2, y2], plus a one-line rationale mentioning neighboring elements.

Return STRICT JSON: {{"regions": [{{"box": [x1,y1,x2,y2], "rationale": "..."}}]}}
If the target is not present, return {{"regions": []}}.

Image size: {width}x{height} pixels.
Instruction: {instruction}
"""

RESULT_CHECK = """\
You are given a cropped screenshot with one element marked by a red box. Decide
whether the marked element matches the target described in my instruction.

Steps:
1. Briefly describe the visible content.
2. Choose exactly one:
   - is_target: the marked element IS the target.
   - target_elsewhere: not the target, but the target IS visible in this view.
   - target_not_found: not the target, and it is NOT in this view at all.

Return STRICT JSON: {{"result": "is_target|target_elsewhere|target_not_found"}}

Instruction: {instruction}
"""

# Popup classification (hybrid handler, VLM fallback path).
POPUP_CLASSIFY = """\
A dialog may be blocking the screen. Read its title, body text, and buttons, then
classify it as EXACTLY one of:
- save_confirmation: asks whether to save changes (buttons like Save/Don't Save/Cancel)
- overwrite_prompt: warns a file already exists and asks to replace (Yes/No)
- path_error: says the path or filename is invalid (OK)
- none: no blocking dialog is present
- unknown: a dialog is present but matches none of the above

Return STRICT JSON: {{"type": "...", "summary": "one short sentence"}}
"""
