from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


INPUT_FILE = Path("requirement_stories.txt")
OUTPUT_FILE = Path("generated_testcases.md")


@dataclass
class LLMConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2


def read_requirements(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Please create it and add user stories."
        )

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"{file_path} is empty. Add at least one user story.")

    return content


def build_prompt(user_stories: str) -> str:
    return f"""
You are a senior QA engineer.
Generate high-quality, practical software test cases from the user stories below.

Requirements for output:
- Output must be valid markdown.
- Group test cases by user story.
- For each test case provide:
  - Test Case ID
  - Title
  - Preconditions
  - Test Data
  - Test Steps (numbered)
  - Expected Result
  - Priority (High/Medium/Low)
  - Type (Positive/Negative/Edge)
- Include both happy path and negative scenarios.
- Include boundary/edge cases where applicable.
- Keep wording clear and executable for manual testing.

User Stories:
{user_stories}
""".strip()


def call_llm(prompt: str, config: LLMConfig) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You generate clear, complete software test cases.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
    }

    req = request.Request(
        url=f"{config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
    except error.HTTPError as http_err:
        details = http_err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP error {http_err.code}: {details}") from http_err
    except error.URLError as url_err:
        raise RuntimeError(f"Could not reach LLM endpoint: {url_err.reason}") from url_err

    try:
        return parsed["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as parse_err:
        raise RuntimeError(f"Unexpected LLM response format: {parsed}") from parse_err


def write_output(content: str, file_path: Path) -> None:
    file_path.write_text(content, encoding="utf-8")


def get_config_from_env() -> LLMConfig:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()

    if not api_key:
        raise EnvironmentError(
            "LLM_API_KEY is not set. Configure it in your environment or PyCharm run configuration."
        )

    return LLMConfig(api_key=api_key, model=model, base_url=base_url)


def main() -> int:
    try:
        requirements_text = read_requirements(INPUT_FILE)
        prompt = build_prompt(requirements_text)
        config = get_config_from_env()

        print("Calling LLM to generate test cases...")
        generated = call_llm(prompt, config)
        write_output(generated, OUTPUT_FILE)

        print(f"Done. Test cases written to: {OUTPUT_FILE}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
