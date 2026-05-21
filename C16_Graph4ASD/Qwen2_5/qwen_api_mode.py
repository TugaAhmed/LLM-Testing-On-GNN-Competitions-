import json
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration  
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
PROVIDER = "nscale" 
TEMPERATURE = 0
TOP_P = 0.99
MAX_OUTPUT_TOKENS = 4000
REPAIR_LIMIT = 5
TIMEOUT_SEC = 3600

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent                   
competition_DIR = SCRIPT_DIR.parent / "Graph4ASD-Challenge"              
PYTHON_EXEC = r"C:\Users\Tuga\Desktop\LLM test\venv1\Scripts\python.exe"
PROMPT_PATH = SCRIPT_DIR / "prompt_filled.md"
RUN_DIR = SCRIPT_DIR / "run_qwen_api"


def extract_block(text: str, tag: str) -> str:
    """
    Extract content from <tag>...</tag>.
    Falls back to fenced code blocks if tags are missing
    (some models skip XML tags and wrap code in ``` instead).
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if tag == "code":
            fenced = re.search(r"^```(?:python)?\s*(.*?)```$", content, re.DOTALL)
            if fenced:
                return fenced.group(1).strip()
        return content

    if tag == "code":
        fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            return fence.group(1).strip()

    if tag == "plan":
        plan_match = re.search(
            r"(?:^|\n)(?:#+\s*)?(?:Plan|Approach)\s*:?\s*(.*?)(?=```|<code>|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if plan_match and plan_match.group(1).strip():
            return plan_match.group(1).strip()

    raise ValueError(f"Missing <{tag}> block in model response")


def write_environment(run_dir: Path):
    """Save platform + pip freeze for reproducibility."""
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"model: {MODEL}",
        f"provider: {PROVIDER}",
        f"working_directory: {competition_DIR}",
    ]
    pip = subprocess.run(
        ["python", "-m", "pip", "freeze"],
        capture_output=True, text=True, check=False,
    )
    lines += ["\n[pip freeze]", pip.stdout]
    (run_dir / "environment.txt").write_text("\n".join(lines) , encoding="utf-8")


def validate_submission(competition_DIR: Path) -> str:
    """
    Check that submission.csv was written correctly.
    competition requires:  id,y_pred columns, 153 rows (one per test graph).
    """
    csv_path = competition_DIR / "submission.csv"
    csv_path2 = competition_DIR / "submissions" / "submission.csv"
    if not csv_path.exists():
        if not csv_path2.exists():
            return "FAIL: submission.csv not found"
        csv_path = csv_path2

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "FAIL: submission.csv is empty"

    header = lines[0].strip()
    if header != "id,y_pred":
        return f"FAIL: wrong header '{header}', expected 'id,y_pred'"

    n_rows = len(lines) - 1  # exclude header
    if n_rows != 153:
        return f"FAIL: expected 153 prediction rows, got {n_rows}"

    return f"OK: submission.csv has {n_rows} rows with correct header"


def call_llm(client: InferenceClient, messages: list) -> str:
    """Single API call to Hugging Face Inference API (OpenAI-compatible)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""


def main():
    if "HF_API_KEY" not in os.environ:
        raise RuntimeError(
            "HF_API_KEY is not set.\n"
            "Get your token at https://huggingface.co/settings/tokens and export it:\n"
            "  export HF_API_KEY=your_token_here"
        )

    if not competition_DIR.exists():
        raise FileNotFoundError(f"competition competition folder not found at: {competition_DIR}")

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"prompt_filled.md not found at: {PROMPT_PATH}")

    RUN_DIR.mkdir(exist_ok=True)

    # Save config for reproducibility
    config = {
        "mode": "api_only",
        "provider": PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "repair_limit": REPAIR_LIMIT,
        "timeout_sec": TIMEOUT_SEC,
        "prompt_path": str(PROMPT_PATH),
        "competition_DIR": str(competition_DIR),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_environment(RUN_DIR)

    # Copy prompt snapshot into run folder
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    (RUN_DIR / "prompt_filled.md").write_text(prompt_text, encoding="utf-8")

    # Build initial message list
    messages = [{"role": "user", "content": prompt_text}]


    client = InferenceClient(
    provider="nscale",        # routes through nscale, billed via HF credits
    api_key=os.environ["HF_API_KEY"],
)

    success = False
    repairs_used = 0
    total_start = time.time()

    for attempt in range(REPAIR_LIMIT + 1):
        print(f"\n=== Hugging Face attempt {attempt} | model={MODEL} ===")

        # --- Call LLM ---
        text = call_llm(client, messages)
        (RUN_DIR / f"attempt_{attempt}_raw_response.txt").write_text(text, encoding="utf-8")

        # --- Parse plan and code ---
        try:
            plan = extract_block(text, "plan")
            code = extract_block(text, "code")
        except Exception as e:
            parse_error = str(e)
            print(f"Parse error: {parse_error}")
            (RUN_DIR / f"attempt_{attempt}_parse_error.txt").write_text(parse_error, encoding="utf-8")

            # Ask LLM to re-format
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    "Your response could not be parsed.\n\n"
                    f"Error: {parse_error}\n\n"
                    "Please return exactly:\n"
                    "<plan>\n...\n</plan>\n\n"
                    "<code>\n...\n</code>"
                ),
            })
            repairs_used = attempt + 1
            continue

        # Save per-attempt artifacts
        (RUN_DIR / f"attempt_{attempt}_plan.md").write_text(plan, encoding="utf-8")
        (RUN_DIR / f"attempt_{attempt}_solution.py").write_text(code, encoding="utf-8")
        (RUN_DIR / "plan.md").write_text(plan, encoding="utf-8")
        (RUN_DIR / "solution.py").write_text(code, encoding="utf-8")

        # Write solution.py into competition root so `python solution.py` works
        solution_in_competition = competition_DIR / "solution.py"
        solution_in_competition.write_text(code, encoding="utf-8")

        # --- Execute solution.py from competition root ---
        print(f"Running solution.py from {competition_DIR} ...")
        start = time.time()
        result = subprocess.run(
            [PYTHON_EXEC, "solution.py"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
            cwd=str(competition_DIR),              # run from competition repo root
        )
        elapsed = time.time() - start

        execution_log = (
            f"attempt={attempt}\n"
            f"returncode={result.returncode}\n"
            f"elapsed_sec={elapsed:.2f}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}\n"
        )
        (RUN_DIR / f"attempt_{attempt}_execution_log.txt").write_text(execution_log, encoding="utf-8")
        (RUN_DIR / "execution_log.txt").write_text(execution_log, encoding="utf-8")
        print(execution_log[:500])

        # --- Validate submission.csv ---
        validation_log = validate_submission(competition_DIR)
        (RUN_DIR / f"attempt_{attempt}_validation_log.txt").write_text(validation_log, encoding="utf-8")
        (RUN_DIR / "validation_log.txt").write_text(validation_log, encoding="utf-8")
        print(f"Validation: {validation_log}")

        # --- Success? ---
        if result.returncode == 0 and validation_log.startswith("OK"):
            print("Success: submission.csv written correctly.")
            shutil.copyfile(competition_DIR / "submission.csv", RUN_DIR / "submission.csv")
            shutil.copyfile(solution_in_competition, RUN_DIR / "final_solution.py")
            success = True
            repairs_used = attempt
            break

        # --- Repair: send traceback back ---
        if attempt < REPAIR_LIMIT:
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    "The script you produced failed.\n\n"
                    f"Return code: {result.returncode}\n\n"
                    f"Stdout:\n{result.stdout}\n\n"
                    f"Stderr / traceback:\n{result.stderr}\n\n"
                    f"Validation: {validation_log}\n\n"
                    "Please return a revised <plan> and <code>. "
                    "Address the error directly; do not restart from scratch unless necessary."
                ),
            })
        repairs_used = attempt + 1

    total_elapsed = time.time() - total_start

    # --- Summary CSV ---
    summary = (
        "mode,provider,model,success,repairs_used,temperature,top_p,"
        "max_output_tokens,total_elapsed_sec\n"
        f"api_only,{PROVIDER},{MODEL},{int(success)},{repairs_used},"
        f"{TEMPERATURE},{TOP_P},{MAX_OUTPUT_TOKENS},{total_elapsed:.2f}\n"
    )
    (RUN_DIR / "run_summary.csv").write_text(summary, encoding="utf-8")
    print(f"\nTotal elapsed: {total_elapsed:.1f}s | success={success} | repairs={repairs_used}")

    if not success:
        raise RuntimeError(
            f"Run failed after {REPAIR_LIMIT} repair attempts. "
            f"Check {RUN_DIR} for logs."
        )


if __name__ == "__main__":
    main()
    

# import json
# import os
# import platform
# import re
# import shutil
# import subprocess
# import time
# from datetime import datetime
# from pathlib import Path
# from openai import OpenAI

# from huggingface_hub import InferenceClient

# # ---------------------------------------------------------------------------
# # Configuration  
# # ---------------------------------------------------------------------------

# # MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
# # # MODEL    = "Qwen/Qwen2.5-72B-Instruct"
# # PROVIDER = "nscale"  # routes through nscale, billed via HF credits. Alternative: "huggingface" for direct API calls (not recommended due to rate limits and lack of retry logic).
# MODEL    = "Qwen/Qwen2.5-Coder-32B-Instruct"
# PROVIDER = "nscale"
# TEMPERATURE = 0
# TOP_P = 0.99
# MAX_OUTPUT_TOKENS = 4000
# REPAIR_LIMIT = 5
# TIMEOUT_SEC = 3600

# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------
# SCRIPT_DIR = Path(__file__).parent                   
# competition_DIR = SCRIPT_DIR.parent / "NetLinkArena"              
# PYTHON_EXEC = r"C:\Users\Tuga\Desktop\LLM test\venv1\Scripts\python.exe"
# PROMPT_PATH = SCRIPT_DIR / "prompt_filled.md"
# RUN_DIR = SCRIPT_DIR / "run_qwen_api"


# def extract_block(text: str, tag: str) -> str:
#     """
#     Extract content from <tag>...</tag>.
#     Falls back to fenced code blocks if tags are missing
#     (some models skip XML tags and wrap code in ``` instead).
#     """
#     match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
#     if match:
#         content = match.group(1).strip()
#         if tag == "code":
#             fenced = re.search(r"^```(?:python)?\s*(.*?)```$", content, re.DOTALL)
#             if fenced:
#                 return fenced.group(1).strip()
#         return content

#     if tag == "code":
#         fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
#         if fence:
#             return fence.group(1).strip()

#     if tag == "plan":
#         plan_match = re.search(
#             r"(?:^|\n)(?:#+\s*)?(?:Plan|Approach)\s*:?\s*(.*?)(?=```|<code>|$)",
#             text,
#             re.DOTALL | re.IGNORECASE,
#         )
#         if plan_match and plan_match.group(1).strip():
#             return plan_match.group(1).strip()

#     raise ValueError(f"Missing <{tag}> block in model response")


# def write_environment(run_dir: Path):
#     """Save platform + pip freeze for reproducibility."""
#     lines = [
#         f"timestamp: {datetime.now().isoformat()}",
#         f"platform: {platform.platform()}",
#         f"python: {platform.python_version()}",
#         f"model: {MODEL}",
#         f"provider: {PROVIDER}",
#         f"working_directory: {competition_DIR}",
#     ]
#     pip = subprocess.run(
#         ["python", "-m", "pip", "freeze"],
#         capture_output=True, text=True, check=False,
#     )
#     lines += ["\n[pip freeze]", pip.stdout]
#     (run_dir / "environment.txt").write_text("\n".join(lines) , encoding="utf-8")


# def validate_submission(competition_DIR: Path) -> str:
#     """
#     Check that submission.csv was written correctly.
#     competition requires:  id,y_pred columns, 1822 rows (one per test graph).
#     """
#     csv_path = competition_DIR / "submission.csv"
#     csv_path2 = competition_DIR / "submissions" / "submission.csv"
#     if not csv_path.exists():
#         if not csv_path2.exists():
#             return "FAIL: submission.csv not found"
#         csv_path = csv_path2

#     lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
#     if not lines:
#         return "FAIL: submission.csv is empty"

#     header = lines[0].strip()
#     if header != "id,y_pred":
#         return f"FAIL: wrong header '{header}', expected 'id,y_pred'"

#     n_rows = len(lines) - 1  # exclude header
#     if n_rows != 1822:
#         return f"FAIL: expected 1822 prediction rows, got {n_rows}"

#     return f"OK: submission.csv has {n_rows} rows with correct header"


# def call_llm(client, messages, retries=3, backoff=15):
#     for attempt in range(retries):
#         try:
#             response = client.chat.completions.create(
#                 model="Qwen/Qwen2.5-Coder-32B-Instruct",
#                 messages=messages,
#                 temperature=TEMPERATURE,
#                 top_p=TOP_P,
#                 max_tokens=MAX_OUTPUT_TOKENS,
#             )
#             return response.choices[0].message.content or ""
#         except Exception as e:
#             print(f"Attempt {attempt+1} failed: {str(e)[:80]}")
#             if attempt < retries - 1:
#                 time.sleep(backoff * (attempt + 1))
#             else:
#                 raise

# def main():

#     if "NSCALE_API_KEY" not in os.environ:
#         raise RuntimeError(
#         "NSCALE_API_KEY is not set.\n"
#         "Get your token at https://console.nscale.com and run:\n"
#         "  $env:NSCALE_API_KEY = 'your_token_here'"
#     )

#     if not competition_DIR.exists():
#         raise FileNotFoundError(f"competition competition folder not found at: {competition_DIR}")

#     if not PROMPT_PATH.exists():
#         raise FileNotFoundError(f"prompt_filled.md not found at: {PROMPT_PATH}")

#     RUN_DIR.mkdir(exist_ok=True)

#     # Save config for reproducibility
#     config = {
#         "mode": "api_only",
#         "provider": PROVIDER,
#         "model": MODEL,
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P,
#         "max_output_tokens": MAX_OUTPUT_TOKENS,
#         "repair_limit": REPAIR_LIMIT,
#         "timeout_sec": TIMEOUT_SEC,
#         "prompt_path": str(PROMPT_PATH),
#         "competition_DIR": str(competition_DIR),
#     }
#     (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
#     write_environment(RUN_DIR)

#     # Copy prompt snapshot into run folder
#     prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
#     (RUN_DIR / "prompt_filled.md").write_text(prompt_text, encoding="utf-8")

#     # Build initial message list
#     messages = [{"role": "user", "content": prompt_text}]

#     client = OpenAI(
#     api_key=os.environ["NSCALE_API_KEY"],
#     base_url="https://inference.api.nscale.com/v1",  # ✅ was missing .api.
# )

#     success = False
#     repairs_used = 0
#     total_start = time.time()

#     for attempt in range(REPAIR_LIMIT + 1):
#         print(f"\n=== Hugging Face attempt {attempt} | model={MODEL} ===")

#         # --- Call LLM ---
#         text = call_llm(client, messages)
#         (RUN_DIR / f"attempt_{attempt}_raw_response.txt").write_text(text, encoding="utf-8")

#         # --- Parse plan and code ---
#         try:
#             plan = extract_block(text, "plan")
#             code = extract_block(text, "code")
#         except Exception as e:
#             parse_error = str(e)
#             print(f"Parse error: {parse_error}")
#             (RUN_DIR / f"attempt_{attempt}_parse_error.txt").write_text(parse_error, encoding="utf-8")

#             # Ask LLM to re-format
#             messages.append({"role": "assistant", "content": text})
#             messages.append({
#                 "role": "user",
#                 "content": (
#                     "Your response could not be parsed.\n\n"
#                     f"Error: {parse_error}\n\n"
#                     "Please return exactly:\n"
#                     "<plan>\n...\n</plan>\n\n"
#                     "<code>\n...\n</code>"
#                 ),
#             })
#             repairs_used = attempt + 1
#             continue

#         # Save per-attempt artifacts
#         (RUN_DIR / f"attempt_{attempt}_plan.md").write_text(plan, encoding="utf-8")
#         (RUN_DIR / f"attempt_{attempt}_solution.py").write_text(code, encoding="utf-8")
#         (RUN_DIR / "plan.md").write_text(plan, encoding="utf-8")
#         (RUN_DIR / "solution.py").write_text(code, encoding="utf-8")

#         # Write solution.py into competition root so `python solution.py` works
#         solution_in_competition = competition_DIR / "solution.py"
#         solution_in_competition.write_text(code, encoding="utf-8")

#         # --- Execute solution.py from competition root ---
#         print(f"Running solution.py from {competition_DIR} ...")
#         start = time.time()
#         result = subprocess.run(
#             [PYTHON_EXEC, "solution.py"],
#             capture_output=True,
#             text=True,
#             timeout=TIMEOUT_SEC,
#             check=False,
#             cwd=str(competition_DIR),              # run from competition repo root
#         )
#         elapsed = time.time() - start

#         execution_log = (
#             f"attempt={attempt}\n"
#             f"returncode={result.returncode}\n"
#             f"elapsed_sec={elapsed:.2f}\n\n"
#             f"STDOUT:\n{result.stdout}\n\n"
#             f"STDERR:\n{result.stderr}\n"
#         )
#         (RUN_DIR / f"attempt_{attempt}_execution_log.txt").write_text(execution_log, encoding="utf-8")
#         (RUN_DIR / "execution_log.txt").write_text(execution_log, encoding="utf-8")
#         print(execution_log[:500])

#         # --- Validate submission.csv ---
#         validation_log = validate_submission(competition_DIR)
#         (RUN_DIR / f"attempt_{attempt}_validation_log.txt").write_text(validation_log, encoding="utf-8")
#         (RUN_DIR / "validation_log.txt").write_text(validation_log, encoding="utf-8")
#         print(f"Validation: {validation_log}")

#         # --- Success? ---
#         if result.returncode == 0 and validation_log.startswith("OK"):
#             print("Success: submission.csv written correctly.")
#             shutil.copyfile(competition_DIR / "submission.csv", RUN_DIR / "submission.csv")
#             shutil.copyfile(solution_in_competition, RUN_DIR / "final_solution.py")
#             success = True
#             repairs_used = attempt
#             break

#         # --- Repair: send traceback back ---
#         if attempt < REPAIR_LIMIT:
#             messages.append({"role": "assistant", "content": text})
#             messages.append({
#                 "role": "user",
#                 "content": (
#                     "The script you produced failed.\n\n"
#                     f"Return code: {result.returncode}\n\n"
#                     f"Stdout:\n{result.stdout}\n\n"
#                     f"Stderr / traceback:\n{result.stderr}\n\n"
#                     f"Validation: {validation_log}\n\n"
#                     "Please return a revised <plan> and <code>. "
#                     "Address the error directly; do not restart from scratch unless necessary."
#                 ),
#             })
#         repairs_used = attempt + 1

#     total_elapsed = time.time() - total_start

#     # --- Summary CSV ---
#     summary = (
#         "mode,provider,model,success,repairs_used,temperature,top_p,"
#         "max_output_tokens,total_elapsed_sec\n"
#         f"api_only,{PROVIDER},{MODEL},{int(success)},{repairs_used},"
#         f"{TEMPERATURE},{TOP_P},{MAX_OUTPUT_TOKENS},{total_elapsed:.2f}\n"
#     )
#     (RUN_DIR / "run_summary.csv").write_text(summary, encoding="utf-8")
#     print(f"\nTotal elapsed: {total_elapsed:.1f}s | success={success} | repairs={repairs_used}")

#     if not success:
#         raise RuntimeError(
#             f"Run failed after {REPAIR_LIMIT} repair attempts. "
#             f"Check {RUN_DIR} for logs."
#         )


# if __name__ == "__main__":
#     main()


















