"""
Prompt Writer

Persists generated debate prompts to a JSON file.

Replaces _write_generated_prompts() in prompt_generator.py, which wrote
a Python source file:

    # Old — wrote generated_prompts.py with triple-quoted strings:
    content = f'''
    ATTACKER_PROMPT = """ """
    '''
    # Required a manual escape() nested function for triple quotes.
    # Required importlib.util dynamic import to read back.

    # New — writes generated_prompts.json:
    json.dump({"attacker": "...", "defender": "...", "judge": "..."}, f)
    # Read back with a simple json.load().

This eliminates:
  • The code-generation smell (writing Python source as a string).
  • The triple-quote escape hack.
  • The importlib dynamic import in both PromptGenerator and PolicyDebate.
"""

from __future__ import annotations

import json
import time


class PromptWriter:
    """
    Writes generated debate prompts to a JSON file.

    Usage
    -----
        writer = PromptWriter("generated_prompts.json")
        writer.write(prompts, meta)
    """

    def __init__(self, output_path: str = "generated_prompts.json") -> None:
        """
        Args:
            output_path: Path to write the JSON prompts file.
        """
        self._path = output_path

    def write(self, prompts: dict[str, str], meta: dict) -> None:
        """
        Write generated prompts and policy metadata to JSON.

        Args:
            prompts: Dict with keys "attacker", "defender", "judge".
            meta:    Policy metadata dict with title, domain,
                     category_summaries (used to derive category names).
        """
        payload = {
            # Generation metadata
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "policy_title": meta.get("title", "Policy Document"),
            "policy_domain": meta.get("domain", ""),
            "categories": list(meta.get("category_summaries", {}).keys()),
            # Role prompts
            "attacker": prompts["attacker"],
            "defender": prompts["defender"],
            "judge": prompts["judge"],
        }

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"  💾 Prompts written to: {self._path}")
