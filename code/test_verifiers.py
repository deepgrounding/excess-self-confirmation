"""Unit smoke tests for verifiers (no MLX)."""
from __future__ import annotations

from verifiers import code_correct, math_correct, normalize_math_answer, run_code_tests


def test_math():
    assert normalize_math_answer("1,234") == normalize_math_answer("1234")
    assert math_correct("stuff\n#### 42\n", "42") == 1.0
    assert math_correct("#### 41", "42") == 0.0
    print("math OK")


def test_code():
    cand = "def add(a, b):\n    return a + b\n"
    tests = "def check(fn):\n    assert fn(1,2)==3\n    assert fn(0,0)==0\n"
    assert run_code_tests(cand, tests, entry_point="add") is True
    bad = "def add(a, b):\n    return a - b\n"
    assert run_code_tests(bad, tests, entry_point="add") is False
    text = "Here is code:\n```python\ndef add(a,b):\n    return a+b\n```\n"
    assert code_correct(text, tests, entry_point="add") == 1.0
    print("code OK")


if __name__ == "__main__":
    test_math()
    test_code()
    print("ALL PASS")
