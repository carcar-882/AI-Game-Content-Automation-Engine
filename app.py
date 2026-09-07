from __future__ import annotations

import io
import json
import re
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_CONNECT_TIMEOUT = 5
OLLAMA_MODEL_TIMEOUT = 300

GAME_TYPES = [
    "Puzzle",
    "Matching Cards",
    "Memory Game",
    "Sequencing",
    "Pattern / Logic",
    "Custom Game Template",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Game Content Automation Engine",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #111827;
        color: #f9fafb;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1120;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .hero {
        padding: 28px 10px;
        text-align: center;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: #1f2937;
        border: 1px solid #374151;
        color: #60a5fa;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 16px;
    }

    .pipeline-card,
    .section-card,
    .metric-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 20px;
    }

    .metric-card {
        text-align: center;
    }

    .metric-label {
        color: #9ca3af;
        font-size: 13px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 700;
    }

    textarea {
        border-radius: 12px !important;
    }

    hr {
        border-color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NEW CHAT
# ============================================================

def reset_chat() -> None:
    """Clear the current pipeline and start a fresh chat/session."""
    for key in [
        "content",
        "scenes",
        "approved",
        "selected_games",
        "game_outputs",
        "validation",
        "raw_input",
        "last_error",
        "generated_png",
        "generated_jpeg",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "content": {},
        "scenes": {},
        "approved": False,
        "selected_games": [],
        "game_outputs": {},
        "validation": {},
        "raw_input": "",
        "last_error": "",
        "generated_png": None,
        "generated_jpeg": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# OLLAMA CLIENT
# ============================================================

def check_ollama() -> tuple[bool, str]:
    """Check whether the local Ollama server is reachable."""
    try:
        response = requests.get(
            OLLAMA_BASE_URL,
            timeout=OLLAMA_CONNECT_TIMEOUT,
        )

        if response.ok:
            return True, "Ollama is running."

        return False, f"Ollama returned HTTP {response.status_code}."

    except requests.RequestException as exc:
        return False, f"Cannot connect to Ollama: {exc}"


@st.cache_data(ttl=10, show_spinner=False)
def get_models() -> list[str]:
    """Return locally installed Ollama model names."""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()

        models = []
        for item in data.get("models", []):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                models.append(name.strip())

        return sorted(set(models))

    except (requests.RequestException, ValueError, TypeError):
        return []


def ask_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Send a structured JSON request to Ollama.

    The function retries transient failures and validates that the
    final response is a JSON object.
    """
    if not model or not model.strip():
        raise ValueError("Please select or enter an Ollama model.")

    payload = {
        "model": model.strip(),
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(temperature),
        },
    }

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                json=payload,
                timeout=OLLAMA_MODEL_TIMEOUT,
            )

            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{model}' was not found. "
                    f"Run: ollama pull {model}"
                )

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    "Ollama returned a non-JSON HTTP response."
                ) from exc

            message = data.get("message")
            if not isinstance(message, dict):
                raise RuntimeError(
                    "Ollama response has no valid message object."
                )

            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Ollama returned an empty model response.")

            return extract_json(content)

        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc

            if attempt < max_retries:
                continue

    raise RuntimeError(
        f"Ollama request failed after {max_retries + 1} attempts: {last_error}"
    )


# ============================================================
# JSON + VALIDATION UTILITIES
# ============================================================

def extract_json(text: str) -> dict[str, Any]:
    """Safely extract a JSON object from an Ollama response."""
    if not text or not text.strip():
        raise ValueError("Ollama returned an empty response.")

    cleaned = text.strip()

    # Remove common markdown fences if a model ignores the JSON-only rule.
    cleaned = re.sub(r"^\s*```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise ValueError("Ollama response must be a JSON object.")
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON object in Ollama response.")

    candidate = cleaned[start:end + 1]

    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from Ollama: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("Extracted JSON is not an object.")

    return result


def require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array.")
    return value


def validate_content_structure(content: dict[str, Any]) -> list[str]:
    """Validate the output of the content engine."""
    errors: list[str] = []

    for field in [
        "title",
        "subject",
        "age_group",
        "difficulty",
        "summary",
    ]:
        if (
            not isinstance(content.get(field), str)
            or not content[field].strip()
        ):
            errors.append(f"Content field '{field}' is missing or empty.")

    for field in [
        "learning_objectives",
        "concepts",
        "objects",
        "actions",
        "relationships",
        "keywords",
    ]:
        if not isinstance(content.get(field), list):
            errors.append(f"Content field '{field}' must be an array.")

    concept_ids = [
        item.get("id")
        for item in content.get("concepts", [])
        if isinstance(item, dict)
    ]
    if len(concept_ids) != len(set(x for x in concept_ids if x)):
        errors.append("Concept IDs must be unique.")

    object_ids = [
        item.get("id")
        for item in content.get("objects", [])
        if isinstance(item, dict)
    ]
    if len(object_ids) != len(set(x for x in object_ids if x)):
        errors.append("Object IDs must be unique.")

    return errors


def validate_scene_structure(scenes: dict[str, Any]) -> list[str]:
    """Validate the output of the scene engine."""
    errors: list[str] = []

    if not isinstance(scenes.get("normalized_title"), str):
        errors.append("Missing normalized_title.")

    scene_list = scenes.get("scenes")
    if not isinstance(scene_list, list) or not scene_list:
        errors.append("No scenes generated.")
        return errors

    ids: list[str] = []

    for index, scene in enumerate(scene_list, start=1):
        if not isinstance(scene, dict):
            errors.append(f"Scene {index} is not an object.")
            continue

        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            errors.append(f"Scene {index} is missing scene_id.")
        else:
            ids.append(scene_id)

        if (
            not isinstance(scene.get("title"), str)
            or not scene["title"].strip()
        ):
            errors.append(f"Scene {index} is missing title.")

        if not isinstance(scene.get("description"), str):
            errors.append(f"Scene {index} has an invalid description.")

        if not isinstance(scene.get("sequence_order"), int):
            errors.append(f"Scene {index} has invalid sequence_order.")

        for field in ["concepts", "objects", "actions", "facts"]:
            if not isinstance(scene.get(field), list):
                errors.append(
                    f"Scene {index} field '{field}' must be an array."
                )

    if len(ids) != len(set(ids)):
        errors.append("Scene IDs must be unique.")

    return errors


# ============================================================
# CONTENT ENGINE
# ============================================================

def content_engine(
    raw_text: str,
    input_type: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:

    system_prompt = """
You are the Content Intelligence Engine of an educational game
automation platform.

Your task is to convert source educational content into accurate,
structured, machine-readable JSON.

Rules:
- Return JSON only.
- Do not return Markdown.
- Do not explain your answer.
- Do not invent unsupported facts.
- Preserve the meaning of the source.
- Use concise educational language.
- Every array must contain valid JSON values.
"""

    user_prompt = f"""
INPUT TYPE:
{input_type}

SOURCE CONTENT:
{raw_text}

Return exactly this JSON structure:

{{
  "title": "",
  "subject": "",
  "age_group": "",
  "difficulty": "",
  "summary": "",
  "learning_objectives": [],
  "concepts": [
    {{
      "id": "concept_001",
      "name": "",
      "description": ""
    }}
  ],
  "objects": [
    {{
      "id": "object_001",
      "name": "",
      "description": ""
    }}
  ],
  "actions": [
    {{
      "id": "action_001",
      "name": "",
      "description": ""
    }}
  ],
  "relationships": [
    {{
      "source": "",
      "relation": "",
      "target": ""
    }}
  ],
  "keywords": []
}}
"""

    result = ask_ollama(
        system_prompt,
        user_prompt,
        model,
        temperature,
    )

    errors = validate_content_structure(result)
    if errors:
        raise ValueError("Content validation failed: " + " | ".join(errors))

    return result


# ============================================================
# SCENE ENGINE
# ============================================================

def scene_engine(
    content: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    system_prompt = """
You are the Scene Structuring Engine.

Convert structured educational content into independent,
game-ready scenes.

Rules:
- Return JSON only.
- Do not invent unrelated facts.
- Preserve source meaning.
- Each scene must have a unique scene_id.
- sequence_order must start at 1 and increase by 1.
"""

    user_prompt = f"""
CONTENT:
{json.dumps(content, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "normalized_title": "",
  "content_type": "",
  "scenes": [
    {{
      "scene_id": "scene_001",
      "title": "",
      "description": "",
      "sequence_order": 1,
      "concepts": [],
      "objects": [],
      "actions": [],
      "facts": [],
      "difficulty": ""
    }}
  ]
}}

Rules:
1. Start sequence_order at 1.
2. Use unique scene IDs.
3. Preserve source meaning.
4. Do not invent unrelated facts.
5. Make each scene useful for educational game generation.
"""

    result = ask_ollama(
        system_prompt,
        user_prompt,
        model,
        temperature,
    )

    errors = validate_scene_structure(result)
    if errors:
        raise ValueError("Scene validation failed: " + " | ".join(errors))

    return result


# ============================================================
# GAME GENERATORS
# ============================================================

def puzzle_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Generate an educational puzzle game from the supplied scenes.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Puzzle",
  "title": "",
  "instructions": "",
  "difficulty": "",
  "questions": [
    {{
      "id": "puzzle_001",
      "question": "",
      "options": [],
      "correct_answer": "",
      "explanation": "",
      "scene_id": ""
    }}
  ]
}}

Generate 5 questions.

Rules:
- correct_answer MUST exactly match one of options.
- scene_id MUST reference an existing scene.
- Do not invent facts.
"""

    return ask_ollama(
        "You are an educational puzzle generator. Return JSON only.",
        prompt,
        model,
        temperature,
    )


def matching_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Generate an educational matching-card game.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Matching Cards",
  "title": "",
  "instructions": "",
  "pairs": [
    {{
      "pair_id": "pair_001",
      "left": "",
      "right": "",
      "scene_id": ""
    }}
  ]
}}

Generate 8 pairs.
"""

    return ask_ollama(
        "You are a matching game generator. Return JSON only.",
        prompt,
        model,
        temperature,
    )


def memory_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Generate an educational memory-card game.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Memory Game",
  "title": "",
  "instructions": "",
  "cards": [
    {{
      "card_id": "card_001",
      "pair_id": "pair_001",
      "content": "",
      "card_type": "question",
      "scene_id": ""
    }}
  ]
}}

Create 8 matching pairs = 16 cards.
Each pair must have two cards with the same pair_id.
"""

    return ask_ollama(
        "You are a memory game generator. Return JSON only.",
        prompt,
        model,
        temperature,
    )


def sequencing_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Generate educational sequencing games.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Sequencing",
  "title": "",
  "instructions": "",
  "sequences": [
    {{
      "sequence_id": "sequence_001",
      "scene_id": "",
      "items": [
        {{
          "id": "item_001",
          "content": "",
          "correct_position": 1
        }}
      ],
      "explanation": ""
    }}
  ]
}}

Generate 3 sequences.
"""

    return ask_ollama(
        "You are a sequencing game generator. Return JSON only.",
        prompt,
        model,
        temperature,
    )


def logic_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Generate educational pattern and logic challenges.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Pattern / Logic",
  "title": "",
  "instructions": "",
  "challenges": [
    {{
      "challenge_id": "challenge_001",
      "pattern": [],
      "options": [],
      "correct_answer": "",
      "explanation": "",
      "scene_id": ""
    }}
  ]
}}

Generate 5 challenges.
The correct_answer must exactly match one option.
"""

    return ask_ollama(
        "You are a logic game generator. Return JSON only.",
        prompt,
        model,
        temperature,
    )


def custom_game_generator(
    scenes: dict[str, Any],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    prompt = f"""
Design a reusable educational game template.

SOURCE:
{json.dumps(scenes, indent=2, ensure_ascii=False)}

Return exactly:

{{
  "game_type": "Custom Game Template",
  "title": "",
  "instructions": "",
  "template": {{
    "objective": "",
    "player_action": "",
    "content_schema": {{}},
    "validation_rules": [],
    "scoring_rules": [],
    "difficulty_rules": []
  }}
}}

Do not invent educational facts outside the source.
"""

    return ask_ollama(
        "You are a game system architect. Return JSON only.",
        prompt,
        model,
        temperature,
    )


# ============================================================
# ORCHESTRATOR
# ============================================================

def game_orchestrator(
    scenes: dict[str, Any],
    selected_games: list[str],
    model: str,
    temperature: float,
) -> dict[str, Any]:

    generators: dict[str, Callable[..., dict[str, Any]]] = {
        "Puzzle": puzzle_generator,
        "Matching Cards": matching_generator,
        "Memory Game": memory_generator,
        "Sequencing": sequencing_generator,
        "Pattern / Logic": logic_generator,
        "Custom Game Template": custom_game_generator,
    }

    outputs: dict[str, Any] = {}

    for game_type in selected_games:
        generator = generators.get(game_type)

        if generator is None:
            raise ValueError(f"Unsupported game type: {game_type}")

        outputs[game_type] = generator(
            scenes,
            model,
            temperature,
        )

    return outputs


# ============================================================
# GAME VALIDATOR
# ============================================================

def validate_games(games: dict[str, Any]) -> dict[str, Any]:
    """Validate generated game structures and cross-references."""
    result = {
        "status": "PASS",
        "total_games": len(games),
        "passed_games": 0,
        "failed_games": 0,
        "checks": [],
    }

    for game_type, game in games.items():
        errors: list[str] = []

        if not isinstance(game, dict):
            errors.append("Game output is not a JSON object.")
        else:
            if (
                not isinstance(game.get("title"), str)
                or not game["title"].strip()
            ):
                errors.append("Missing title.")

            if (
                not isinstance(game.get("instructions"), str)
                or not game["instructions"].strip()
            ):
                errors.append("Missing instructions.")

            if game_type == "Puzzle":
                questions = game.get("questions")
                if not isinstance(questions, list) or not questions:
                    errors.append("No questions generated.")
                else:
                    ids = []
                    for index, item in enumerate(questions, 1):
                        if not isinstance(item, dict):
                            errors.append(
                                f"Puzzle question {index} is invalid."
                            )
                            continue

                        question_id = item.get("id")
                        if question_id:
                            ids.append(question_id)

                        options = item.get("options")
                        answer = item.get("correct_answer")

                        if not isinstance(options, list) or len(options) < 2:
                            errors.append(
                                f"Puzzle question {index} needs at least 2 "
                                "options."
                            )
                        elif answer not in options:
                            errors.append(
                                f"Puzzle question {index}: correct_answer "
                                "is not in options."
                            )

                        if not item.get("question"):
                            errors.append(
                                f"Puzzle question {index} is missing question "
                                "text."
                            )

                    if len(ids) != len(set(ids)):
                        errors.append("Puzzle question IDs are not unique.")

            elif game_type == "Matching Cards":
                pairs = game.get("pairs")
                if not isinstance(pairs, list) or len(pairs) < 2:
                    errors.append("Insufficient matching pairs.")
                else:
                    pair_ids = [
                        item.get("pair_id")
                        for item in pairs
                        if isinstance(item, dict)
                    ]
                    if len(pair_ids) != len(set(pair_ids)):
                        errors.append("Matching pair IDs are not unique.")

            elif game_type == "Memory Game":
                cards = game.get("cards")
                if not isinstance(cards, list) or len(cards) < 4:
                    errors.append("Insufficient memory cards.")
                else:
                    pair_counts: dict[str, int] = {}
                    for item in cards:
                        if isinstance(item, dict):
                            pair_id = item.get("pair_id")
                            if pair_id:
                                pair_counts[pair_id] = (
                                    pair_counts.get(pair_id, 0) + 1
                                )

                    invalid_pairs = [
                        pair_id
                        for pair_id, count in pair_counts.items()
                        if count != 2
                    ]
                    if invalid_pairs:
                        errors.append(
                            "Every memory pair must contain exactly 2 cards."
                        )

            elif game_type == "Sequencing":
                sequences = game.get("sequences")
                if not isinstance(sequences, list) or not sequences:
                    errors.append("No sequences generated.")
                else:
                    for index, sequence in enumerate(sequences, 1):
                        items = (
                            sequence.get("items")
                            if isinstance(sequence, dict)
                            else None
                        )
                        if not isinstance(items, list) or not items:
                            errors.append(f"Sequence {index} has no items.")
                            continue

                        positions = [
                            item.get("correct_position")
                            for item in items
                            if isinstance(item, dict)
                        ]

                        expected_positions = list(
                            range(1, len(positions) + 1)
                        )
                        if sorted(positions) != expected_positions:
                            errors.append(
                                f"Sequence {index} positions must be "
                                "continuous starting at 1."
                            )

            elif game_type == "Pattern / Logic":
                challenges = game.get("challenges")
                if not isinstance(challenges, list) or not challenges:
                    errors.append("No logic challenges generated.")
                else:
                    for index, challenge in enumerate(challenges, 1):
                        if not isinstance(challenge, dict):
                            errors.append(
                                f"Logic challenge {index} is invalid."
                            )
                            continue

                        options = challenge.get("options")
                        answer = challenge.get("correct_answer")

                        if not isinstance(options, list) or len(options) < 2:
                            errors.append(
                                f"Logic challenge {index} needs at least 2 "
                                "options."
                            )
                        elif answer not in options:
                            errors.append(
                                f"Logic challenge {index}: correct_answer "
                                "is not in options."
                            )

            elif game_type == "Custom Game Template":
                template = game.get("template")
                if not isinstance(template, dict):
                    errors.append("Missing custom game template.")
                else:
                    for field in [
                        "objective",
                        "player_action",
                        "content_schema",
                        "validation_rules",
                        "scoring_rules",
                        "difficulty_rules",
                    ]:
                        if field not in template:
                            errors.append(
                                f"Template field '{field}' is missing."
                            )

        if errors:
            result["failed_games"] += 1
            result["status"] = "FAIL"
            result["checks"].append(
                {
                    "game_type": game_type,
                    "status": "FAIL",
                    "errors": errors,
                }
            )
        else:
            result["passed_games"] += 1
            result["checks"].append(
                {
                    "game_type": game_type,
                    "status": "PASS",
                    "errors": [],
                }
            )

    return result


# ============================================================
# FINAL PNG / JPEG RENDERING
# ============================================================

def _font(size: int, bold: bool = False):
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _wrapped_lines(text: str, width: int = 82) -> list[str]:
    result = []
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            result.append("")
        else:
            result.extend(textwrap.wrap(paragraph, width=width) or [""])
    return result


def render_final_game_image(
    games: dict[str, Any],
    content: dict[str, Any],
) -> tuple[bytes, bytes]:
    """Render validated Ollama game output into PNG and JPEG."""
    width = 1600
    margin = 70
    title_font = _font(48, True)
    section_font = _font(30, True)
    body_font = _font(23)
    small_font = _font(18)

    sections = []

    for game_type, game in games.items():
        lines = []
        if not isinstance(game, dict):
            lines.append("Invalid game output.")
            sections.append((game_type, lines))
            continue

        lines.append(str(game.get("title", game_type)))
        if game.get("instructions"):
            lines.append("Instructions: " + str(game["instructions"]))

        if game_type == "Puzzle":
            for item in game.get("questions", [])[:5]:
                if isinstance(item, dict):
                    lines.append("Q: " + str(item.get("question", "")))
                    options = item.get("options", [])
                    if isinstance(options, list):
                        lines.append(
                            "Options: " + " | ".join(map(str, options))
                        )
                    lines.append(
                        "Answer: " + str(item.get("correct_answer", ""))
                    )

        elif game_type == "Matching Cards":
            for item in game.get("pairs", [])[:8]:
                if isinstance(item, dict):
                    lines.append(
                        f"{item.get('left', '')}  ↔  {item.get('right', '')}"
                    )

        elif game_type == "Memory Game":
            for item in game.get("cards", [])[:16]:
                if isinstance(item, dict):
                    lines.append(
                        f"[{item.get('pair_id', '')}] "
                        f"{item.get('content', '')}"
                    )

        elif game_type == "Sequencing":
            for sequence in game.get("sequences", [])[:3]:
                if not isinstance(sequence, dict):
                    continue
                lines.append(
                    "Sequence: " + str(sequence.get("sequence_id", ""))
                )
                items = sequence.get("items", [])
                if isinstance(items, list):
                    valid_items = [x for x in items if isinstance(x, dict)]
                    valid_items.sort(
                        key=lambda x: x.get("correct_position", 999)
                    )
                    for item in valid_items:
                        lines.append(
                            f"{item.get('correct_position', '')}. "
                            f"{item.get('content', '')}"
                        )

        elif game_type == "Pattern / Logic":
            for item in game.get("challenges", [])[:5]:
                if isinstance(item, dict):
                    lines.append("Pattern: " + str(item.get("pattern", [])))
                    options = item.get("options", [])
                    if isinstance(options, list):
                        lines.append(
                            "Options: " + " | ".join(map(str, options))
                        )
                    lines.append(
                        "Answer: " + str(item.get("correct_answer", ""))
                    )

        elif game_type == "Custom Game Template":
            template = game.get("template", {})
            if isinstance(template, dict):
                for key in [
                    "objective",
                    "player_action",
                    "validation_rules",
                    "scoring_rules",
                    "difficulty_rules",
                ]:
                    if key in template:
                        lines.append(
                            f"{key.replace('_', ' ').title()}: {template[key]}"
                        )

        sections.append((game_type, lines))

    estimated_lines = 0
    for _, lines in sections:
        for line in lines:
            estimated_lines += max(1, (len(str(line)) // 78) + 1)
        estimated_lines += 2

    height = max(
        1000,
        180 + margin + estimated_lines * 36 + len(sections) * 55,
    )

    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (35, 35, width - 35, 145),
        radius=25,
        fill=(17, 24, 39),
    )

    draw.text(
        (margin, 55),
        "AI GAME OUTPUT",
        font=section_font,
        fill=(255, 255, 255),
    )
    draw.text(
        (margin, 94),
        str(content.get("title", "Educational Game"))[:75],
        font=title_font,
        fill=(255, 255, 255),
    )

    y = 180

    for game_type, lines in sections:
        draw.rounded_rectangle(
            (margin - 15, y, width - margin + 15, y + 62),
            radius=16,
            fill=(229, 231, 235),
        )
        draw.text(
            (margin, y + 14),
            game_type,
            font=section_font,
            fill=(17, 24, 39),
        )
        y += 82

        for line in lines:
            for wrapped in _wrapped_lines(line, 82):
                draw.text(
                    (margin, y),
                    wrapped,
                    font=body_font,
                    fill=(35, 35, 35),
                )
                y += 36
            y += 3

        y += 55

    draw.text(
        (margin, height - 45),
        "Generated locally with Ollama",
        font=small_font,
        fill=(107, 114, 128),
    )

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG", optimize=True)

    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format="JPEG", quality=95, optimize=True)

    return png_buffer.getvalue(), jpeg_buffer.getvalue()


# ============================================================
# EXPORT
# ============================================================

def create_package(
    content: dict[str, Any],
    scenes: dict[str, Any],
    games: dict[str, Any],
    validation: dict[str, Any],
) -> bytes:
    """Create a portable ZIP package containing all pipeline outputs."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.writestr(
            "content.json",
            json.dumps(content, indent=2, ensure_ascii=False),
        )

        archive.writestr(
            "scenes.json",
            json.dumps(scenes, indent=2, ensure_ascii=False),
        )

        archive.writestr(
            "games.json",
            json.dumps(games, indent=2, ensure_ascii=False),
        )

        archive.writestr(
            "validation.json",
            json.dumps(validation, indent=2, ensure_ascii=False),
        )

        archive.writestr(
            "README.txt",
            (
                "GGR Ollama Game Package\n"
                "Generated by AI Game Content Automation Engine\n"
                "AI provider: Ollama (local)\n"
                f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
            ),
        )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.title("🎮 AI Game Engine")
        st.caption("Python + Streamlit + Ollama")

        if st.button("🆕 New Chat", use_container_width=True):
            reset_chat()

        st.markdown("---")

        models = get_models()

        if models:
            if DEFAULT_MODEL in models:
                default_index = models.index(DEFAULT_MODEL)
            else:
                default_index = 0

            model = st.selectbox(
                "Ollama Model",
                models,
                index=default_index,
                help="Models detected from your local Ollama server.",
            )
        else:
            model = st.text_input(
                "Ollama Model",
                value=DEFAULT_MODEL,
                help="Example: qwen2.5:3b",
            )

            st.warning(
                "No local Ollama models were detected. "
                "Make sure Ollama is running and a model is installed."
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )

        st.markdown("---")

        st.markdown(
            """
            ### Pipeline

            📥 Input

            ↓

            🧠 AI Content Engine

            ↓

            🧹 Normalization

            ↓

            🎬 Scene Splitting

            ↓

            👤 Manual Approval

            ↓

            🎮 Game Selection

            ↓

            ⚙️ Automation

            ↓

            ✅ Validation

            ↓

            📦 Export
            """
        )

        st.markdown("---")
        st.caption("AI runs locally through Ollama.")
        st.caption("No OpenAI API is used by this application.")

    return model.strip(), temperature


# ============================================================
# MAIN UI
# ============================================================

def main() -> None:
    model, temperature = render_sidebar()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">LOCAL AI AUTOMATION ENGINE</div>
            <div class="hero-title">AI Game Content Automation Engine</div>
            <div class="hero-subtitle">
                Transform educational content into multiple AI-generated
                game formats using Ollama.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Ollama status
    # --------------------------------------------------------

    ollama_ok, ollama_message = check_ollama()

    if not ollama_ok:
        st.error("🔴 Ollama is not connected.")
        st.warning(ollama_message)

        st.code(
            "ollama list",
            language="powershell",
        )

        st.info(
            "If Ollama is installed, start the Ollama application. "
            "If the server is not running, you can also run: ollama serve"
        )

        st.stop()

    st.success(f"🟢 Ollama Connected — {model}")

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    st.header("1️⃣ Input")

    input_type = st.selectbox(
        "Content Type",
        [
            "Educational Topic",
            "Story",
            "Lesson",
            "Concept",
            "Image Description",
            "Custom Content",
        ],
    )

    raw_input = st.text_area(
        "Enter your content",
        value=st.session_state.raw_input,
        height=220,
        placeholder=(
            "Example:\n"
            "The Solar System consists of the Sun and planets "
            "orbiting around it..."
        ),
    )

    st.session_state.raw_input = raw_input

    if st.button(
        "🚀 Generate Content",
        use_container_width=True,
        type="primary",
    ):
        if not raw_input.strip():
            st.warning("Please enter content.")
        else:
            with st.spinner("Ollama is structuring your content..."):
                try:
                    content = content_engine(
                        raw_input.strip(),
                        input_type,
                        model,
                        temperature,
                    )

                    st.session_state.content = content
                    st.session_state.scenes = {}
                    st.session_state.approved = False
                    st.session_state.selected_games = []
                    st.session_state.game_outputs = {}
                    st.session_state.validation = {}
                    st.session_state.generated_png = None
                    st.session_state.generated_jpeg = None
                    st.session_state.last_error = ""

                    st.success("Content successfully structured by Ollama.")

                except Exception as error:
                    st.session_state.last_error = str(error)
                    st.error(f"AI error: {error}")

    # --------------------------------------------------------
    # Structured content
    # --------------------------------------------------------

    content = st.session_state.content

    if content:
        st.header("2️⃣ Structured Content")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Concepts",
                len(content.get("concepts", [])),
            )

        with col2:
            st.metric(
                "Objects",
                len(content.get("objects", [])),
            )

        with col3:
            st.metric(
                "Learning Objectives",
                len(content.get("learning_objectives", [])),
            )

        with st.expander("View structured JSON", expanded=False):
            st.json(content)

    # --------------------------------------------------------
    # Scene generation
    # --------------------------------------------------------

    if content:
        st.header("3️⃣ Content Normalization + Scene Splitting")

        if st.button(
            "🎬 Generate Game-Ready Scenes",
            use_container_width=True,
        ):
            with st.spinner("Ollama is generating game-ready scenes..."):
                try:
                    scenes = scene_engine(
                        content,
                        model,
                        temperature,
                    )

                    st.session_state.scenes = scenes
                    st.session_state.approved = False
                    st.session_state.selected_games = []
                    st.session_state.game_outputs = {}
                    st.session_state.validation = {}
                    st.session_state.generated_png = None
                    st.session_state.generated_jpeg = None

                    st.success("Scenes generated successfully.")

                except Exception as error:
                    st.error(f"Scene error: {error}")

    # --------------------------------------------------------
    # Scene review
    # --------------------------------------------------------

    scenes = st.session_state.scenes

    if scenes:
        st.header("4️⃣ Manual Review & Approval")

        scene_list = scenes.get("scenes", [])

        st.info(f"{len(scene_list)} scene(s) generated.")

        for scene in scene_list:
            scene_id = scene.get("scene_id", "Scene")
            title = scene.get("title", "Untitled Scene")

            with st.expander(
                f"{scene_id} — {title}",
                expanded=False,
            ):
                st.write(scene.get("description", ""))

                st.write("**Facts:**")
                facts = scene.get("facts", [])

                if facts:
                    for fact in facts:
                        st.write(f"• {fact}")
                else:
                    st.caption("No facts provided.")

                st.write("**Concepts:**")
                st.write(scene.get("concepts", []))

                st.write("**Objects:**")
                st.write(scene.get("objects", []))

                st.write("**Actions:**")
                st.write(scene.get("actions", []))

        approved = st.checkbox(
            "I reviewed and approve these scenes.",
            value=st.session_state.approved,
        )

        st.session_state.approved = approved

    # --------------------------------------------------------
    # Game selection
    # --------------------------------------------------------

    if scenes and st.session_state.approved:
        st.header("5️⃣ Select Game Output Formats")

        selected_games = st.multiselect(
            "Game Formats",
            GAME_TYPES,
            default=st.session_state.selected_games,
        )

        st.session_state.selected_games = selected_games

        if selected_games:
            if st.button(
                "⚙️ Generate Games with Ollama",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner(
                    "Ollama is generating your selected games..."
                ):
                    try:
                        games = game_orchestrator(
                            scenes,
                            selected_games,
                            model,
                            temperature,
                        )

                        st.session_state.game_outputs = games
                        st.session_state.validation = {}
                        st.session_state.generated_png = None
                        st.session_state.generated_jpeg = None

                        st.success(
                            f"{len(games)} game format(s) generated "
                            "successfully."
                        )

                    except Exception as error:
                        st.error(f"Game generation error: {error}")

    # --------------------------------------------------------
    # Game output
    # --------------------------------------------------------

    games = st.session_state.game_outputs

    if games:
        st.header("6️⃣ Generated Game Outputs")

        for game_type, game in games.items():
            with st.expander(
                f"🎮 {game_type}",
                expanded=False,
            ):
                st.json(game)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if games:
        st.header("7️⃣ Quality Validation")

        if st.button(
            "✅ Validate Games",
            use_container_width=True,
        ):
            with st.spinner("Validating generated game structures..."):
                validation = validate_games(games)
                st.session_state.validation = validation

        validation = st.session_state.validation

        if validation:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total",
                    validation["total_games"],
                )

            with col2:
                st.metric(
                    "Passed",
                    validation["passed_games"],
                )

            with col3:
                st.metric(
                    "Failed",
                    validation["failed_games"],
                )

            if validation["status"] == "PASS":
                st.success("✅ All generated games passed validation.")
            else:
                st.error("❌ Some games failed validation.")

            for check in validation["checks"]:
                if check["status"] == "PASS":
                    st.success(f'✅ {check["game_type"]}')
                else:
                    st.error(f'❌ {check["game_type"]}')

                    for error in check["errors"]:
                        st.write(f"• {error}")

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    validation = st.session_state.validation

    if games and validation:
        st.header("8️⃣ Final Output")

        if validation["status"] == "PASS":
            if (
                st.session_state.generated_png is None
                or st.session_state.generated_jpeg is None
            ):
                with st.spinner("Rendering final output as PNG and JPEG..."):
                    try:
                        png_data, jpeg_data = render_final_game_image(
                            games,
                            content,
                        )
                        st.session_state.generated_png = png_data
                        st.session_state.generated_jpeg = jpeg_data
                    except Exception as error:
                        st.error(f"Image rendering error: {error}")

            if (
                st.session_state.generated_png is not None
                and st.session_state.generated_jpeg is not None
            ):
                st.success("Final output is ready as PNG and JPEG.")

                st.image(
                    st.session_state.generated_png,
                    caption="Final Game Output",
                    use_container_width=True,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        "🖼️ Download PNG",
                        data=st.session_state.generated_png,
                        file_name=f"ggr_game_{timestamp}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with col2:
                    st.download_button(
                        "🖼️ Download JPEG",
                        data=st.session_state.generated_jpeg,
                        file_name=f"ggr_game_{timestamp}.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
        else:
            st.warning(
                "Resolve validation errors before creating the final image."
            )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "AI Game Content Automation Engine • "
        "Ollama Local AI • Python • Streamlit"
    )


if __name__ == "__main__":
    main()
