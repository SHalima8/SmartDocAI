"""
llm.py
======
--------------
This module has one responsibility:

    Send a finished prompt to Gemini and return a structured response.

Everything before this step has already happened:

"""

import json
import os
import re
import time
import traceback
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

ENV_PATH = os.path.join(ROOT_DIR, ".env")

print("\n========== ENVIRONMENT DEBUG ==========")
print(f"ROOT_DIR : {ROOT_DIR}")
print(f".env path: {ENV_PATH}")
print(f".env exists? {os.path.exists(ENV_PATH)}")

loaded = load_dotenv(ENV_PATH)

print(f"load_dotenv() returned: {loaded}")

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"GEMINI_API_KEY loaded ✓")
    print(f"Length : {len(api_key)}")
    print(f"First 8 chars : {api_key[:8]}")
    print(f"Last 4 chars  : {api_key[-4:]}")
else:
    print("GEMINI_API_KEY NOT FOUND")

print("=======================================\n")


MODEL_NAME = "gemini-2.5-flash"

_model = None


# ============================================================
# Gemini Initialisation
# ============================================================

def get_model() -> genai.GenerativeModel:
    """
    Initialise Gemini only once.

    Returns
    -------
    GenerativeModel
    """

    print("\nEntering get_model()")

    global _model

    if _model is not None:
        return _model

    api_key = os.getenv("GEMINI_API_KEY")
    print("Checking API key...")

    if not api_key:
        raise RuntimeError(
            "\nGEMINI_API_KEY not found.\n"
            "Create a .env file in the project root:\n\n"
            "GEMINI_API_KEY=your_key_here\n"
        )

    genai.configure(api_key=api_key)
    print("Configuring Gemini...")

    _model = genai.GenerativeModel(MODEL_NAME)
    print("Model object created.")

    print(f"[llm] Gemini model initialised ({MODEL_NAME})")

    return _model

# ============================================================
# Response Parsing
# ============================================================

def extract_json(response_text: str) -> dict[str, Any] | None:
    """
    Extract JSON from Gemini's response.

    Gemini usually follows instructions, but it sometimes wraps
    the JSON inside markdown code fences like:

    ```json
    {
      ...
    }
    ```

    or writes:

    Here is the JSON:
    { ... }

    This function tries several parsing strategies before giving up.
    """

    if not response_text:
        return None

    text = response_text.strip()

    # --------------------------------------------------------
    # Attempt 1
    # Direct JSON parsing
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 2
    # Remove ```json fences
    # --------------------------------------------------------

    cleaned = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Attempt 3
    # Find first {...}
    # --------------------------------------------------------

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if match:

        try:
            return json.loads(match.group())

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# Standard Response Objects
# ============================================================

def success_response(
    parsed: dict[str, Any],
    raw_text: str,
    latency: float,
) -> dict:
    """
    Standard successful response returned by generate_answer().
    """

    return {

        "success": True,

        "answer": parsed.get("answer", ""),

        "confidence": parsed.get(
            "confidence",
            "Unknown"
        ),

        "source_chunk": parsed.get(
            "source_chunk",
            "Unknown"
        ),

        "reasoning": parsed.get(
            "reasoning",
            ""
        ),

        "model": MODEL_NAME,

        "latency": latency,

        "json_parsed": True,

        "raw_response": raw_text,
    }


def raw_response(
    raw_text: str,
    latency: float,
) -> dict:
    """
    Used when Gemini answered successfully but did not
    return valid JSON.

    We still show the answer instead of crashing.
    """

    return {

        "success": True,

        "answer": raw_text,

        "confidence": "Unknown",

        "source_chunk": "Unknown",

        "reasoning": "",

        "model": MODEL_NAME,

        "latency": latency,

        "json_parsed": False,

        "raw_response": raw_text,
    }


def error_response(
    message: str,
    latency: float = 0.0,
) -> dict:
    """
    Returned whenever Gemini cannot be reached or another
    unexpected error occurs.
    """

    return {

        "success": False,

        "answer": "Unable to generate an answer.",

        "confidence": "None",

        "source_chunk": "None",

        "reasoning": "",

        "error": message,

        "model": MODEL_NAME,

        "latency": latency,

        "json_parsed": False,

        "raw_response": "",
    }
# ============================================================
# Gemini Inference
# ============================================================

def generate_answer(prompt: str) -> dict:
    """
    Send a finished prompt to Gemini and return
    a structured response dictionary.

    Parameters
    ----------
    prompt : str
        The final prompt produced by prompt_builder.py.

    Returns
    -------
    dict
        {
            success,
            answer,
            confidence,
            source_chunk,
            reasoning,
            latency,
            model,
            ...
        }
    """

    # --------------------------------------------------------
    # Validate prompt
    # --------------------------------------------------------

    if not prompt or not prompt.strip():

        return error_response(
            "Prompt is empty."
        )

    # --------------------------------------------------------
    # Load Gemini
    # --------------------------------------------------------

    try:

        model = get_model()

    except Exception as e:

        print("\n========== GEMINI INITIALIZATION ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("=================================================\n")

        return error_response(str(e))

    print("\n" + "=" * 60)
    print(" SmartDocAI — Gemini")
    print("=" * 60)

    print(f"Model : {MODEL_NAME}")
    print(f"Prompt length : {len(prompt):,} characters")

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    try:

        start = time.perf_counter()

        print("\nSending request to Gemini...")

        response = model.generate_content(

            prompt,

            generation_config=genai.types.GenerationConfig(

                temperature=0.1,

                max_output_tokens=1024,

            )

        )

        print("Gemini replied successfully.")

        latency = round(
            time.perf_counter() - start,
            3,
        )

    except Exception as e:

      print("\n========== GEMINI API ERROR ==========")
      print(type(e).__name__)
      print(str(e))

      print("\nFull traceback:")
      traceback.print_exc()

      print("======================================\n")

      return error_response(
        message=str(e),
        latency=0.0,
    )

    

    # --------------------------------------------------------
    # Empty response?
    # --------------------------------------------------------

    if not response.text:

        return error_response(

            "Gemini returned an empty response.",

            latency,

        )

    raw_text = response.text.strip()

    print(f"Latency : {latency} sec")
    print(f"Response size : {len(raw_text)} characters")

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    parsed = extract_json(raw_text)

    if parsed:

        print("JSON parsing : SUCCESS")

        return success_response(

            parsed,

            raw_text,

            latency,

        )

    print("JSON parsing : FAILED")
    print("Returning raw text instead.")

    return raw_response(

        raw_text,

        latency,

    )
# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":

    import sys

    from pdf_loader import load_pdf
    from preprocessing import preprocess_document
    from chunking import chunk_document
    from retrieval import retrieve_all
    from prompt_builder import build_prompt_from_comparison


    # --------------------------------------------------------
    # PDF path
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        pdf_path = sys.argv[1]

    else:

        pdf_path = os.path.join(

            ROOT_DIR,

            "data",

            "uploads",

            "attention.pdf",

        )


    print("\n" + "=" * 70)
    print(" SmartDocAI — LLM Demo")
    print("=" * 70)

    # --------------------------------------------------------
    # Build pipeline once
    # --------------------------------------------------------

    try:

        print("\nLoading PDF...")
        document = load_pdf(pdf_path)

        print("\nPreprocessing...")
        preprocess_document(document)

        print("\nChunking...")
        chunks = chunk_document(document)

        print(f"\n✓ {len(chunks)} chunks ready.")

    except Exception as e:

        print(f"\nPipeline failed.\n{e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" Ask questions about the document")
    print(" Type 'quit' to exit.")
    print("=" * 70)


    # --------------------------------------------------------
    # Question loop
    # --------------------------------------------------------

    while True:

        question = input("\nQuestion: ").strip()

        if not question:

            print("Please enter a question.")
            continue

        if question.lower() in [

            "quit",

            "exit",

            "q",

        ]:

            print("\nGoodbye.")
            break

        print("\nRetrieving relevant chunks...")

        comparison = retrieve_all(

            query=question,

            chunks=chunks,

            top_k=3,

        )

        print("Building prompt...")

        prompt_package = build_prompt_from_comparison(

            query=question,

            comparison=comparison,

        )

        print("Sending to Gemini...")

        result = generate_answer(

            prompt_package.prompt,

        )

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(result["answer"])

        print("\n" + "-" * 70)

        print(f"Confidence : {result['confidence']}")
        print(f"Source Chunk: {result['source_chunk']}")
        print(f"Latency     : {result['latency']} sec")
        print(f"Model       : {result['model']}")

        if result.get("reasoning"):

            print("\nReasoning")
            print("-" * 70)
            print(result["reasoning"])

        print("\n" + "=" * 70)
