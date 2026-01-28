# prompt_parser.py
import json
from openai import OpenAI

client = OpenAI()

def parse_prompt_to_plan(prompt):

    system_message = """
You generate ONLY JSON describing audio effects.
Never speak outside JSON.

IMPORTANT RULES:
- If the user asks for multiple effects, include ALL of them.
- Bitcrush is NEVER exclusive unless the user explicitly says "only bitcrush".
- Use reasonable defaults.
- Subtle words mean lighter settings.
- Strong words mean heavier settings.

Allowed effects list:

- autotune
- robotic_autotune
- formant_shift
- pitch_shift
- harmony
- reverb
- sick_verb
- big_verb
- small_room
- space_mod
- chorus
- flanger
- phaser
- distortion
- overdrive
- bitcrush
- compressor
- limiter
- gain
- eq
- hp_filter
- lp_filter
- bp_filter
- notch_filter
- noise_gate
- delay
- hype_delay
- pingpong_delay
- slap_delay
- tempo_tweak
- timestretch
- pitch_correct
- key_detect
- deess
- saturation
- widen
- stereo_spread
- normalize
- reverse
- robot_voice
- vocoder
- megaphone
- telephone
- lofi
- wobble
- vibrato
- tremolo

Return ONLY a JSON array like:

[
  {"effect": "reverb", "params": {"mix": 0.6}},
  {"effect": "bitcrush", "params": {"bit_depth": 6, "sample_rate": 12000}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        content = content.replace("json", "").strip()

    try:
        plan = json.loads(content)

        # ---------------------------------------
        # NATURAL LANGUAGE FIX FOR HARMONIZER
        # ---------------------------------------
        spoken_to_num = {
            "one":1,"two":2,"three":3,"four":4,"five":5,
            "1":1,"2":2,"3":3,"4":4,"5":5
        }

        detected_voices = None
        for word in prompt.lower().split():
            if word in spoken_to_num:
                detected_voices = spoken_to_num[word]
                break

        # If harmonizer/harmony exists, inject voices
        if detected_voices:
            for step in plan:
                if step.get("effect") in ("harmony","harmonizer"):
                    if "params" not in step:
                        step["params"] = {}
                    step["params"]["voices"] = detected_voices

        return plan

    except Exception as e:
        print("❌ PROMPT PARSER FAILED:", e)
        print("❌ RAW CONTENT:", content if 'content' in locals() else 'NO CONTENT')
        raise RuntimeError("Prompt parsing failed")
