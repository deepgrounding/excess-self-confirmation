"""Oracle verifiers for ESC Stage-2 domains (math + code).

Math: normalized exact match (GSM8K / MATH-style #### answers).
Code: subprocess unit-test sandbox (HumanEval / MBPP), 10 s / 512 MB / no network
per Appendix D of draft/main.md.
"""
from __future__ import annotations

import ast
import os
import re
import resource
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


# ----------------------------------------------------------------------
# Math
# ----------------------------------------------------------------------

def normalize_math_answer(s: str | None) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace(",", "")
    t = re.sub(r"\$+", "", t)
    # strip trailing punctuation
    t = t.rstrip(".")
    # try Fraction for 1/2 vs 0.5
    try:
        if "/" in t and re.fullmatch(r"-?\d+/\d+", t):
            return str(float(Fraction(t)))
        return str(float(t))
    except Exception:
        return t.lower()


def extract_math_answer(text: str) -> str | None:
    m = re.findall(r"####\s*([^\n]+)", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?i)final answer\s*[:：]\s*([^\n]+)", text)
    if m:
        return m[-1].strip()
    return None


def math_correct(text: str, gold: str) -> float:
    pred = extract_math_answer(text)
    if pred is None:
        return 0.0
    return 1.0 if normalize_math_answer(pred) == normalize_math_answer(gold) else 0.0


# ----------------------------------------------------------------------
# Code sandbox
# ----------------------------------------------------------------------

def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    if "def " in text:
        return text[text.find("def ") :].strip()
    return None


def _limit_resources() -> None:
    # Soft limits inside the child (best-effort; macOS may ignore some).
    try:
        mem = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass


def run_code_tests(
    candidate_code: str,
    test_code: str,
    entry_point: str | None = None,
    timeout: float = 10.0,
) -> bool:
    """Run candidate + tests in an isolated subprocess.

    Not a hardened jail (no seccomp); suitable for local public-benchmark eval
    under the caller's control — same posture as HumanEval harnesses.
    """
    check = ""
    if entry_point:
        # HumanEval style: check(candidate)
        check = f"\ncheck({entry_point})\n"
    program = candidate_code.rstrip() + "\n\n" + test_code.rstrip() + check
    # Reject obvious network imports at parse time (soft filter)
    try:
        tree = ast.parse(candidate_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                else:
                    names = [node.module.split(".")[0]] if node.module else []
                if any(n in {"socket", "urllib", "requests", "http", "subprocess", "ctypes"} for n in names):
                    return False
    except SyntaxError:
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=td,
                capture_output=True,
                timeout=timeout,
                env=env,
                preexec_fn=_limit_resources if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        return proc.returncode == 0


def code_correct(text: str, test_code: str, entry_point: str | None = None) -> float:
    code = extract_code(text)
    if not code:
        return 0.0
    return 1.0 if run_code_tests(code, test_code, entry_point=entry_point) else 0.0
