# sumospace/benchmarks/fixtures/sample_project/buggy.py
# BENCHMARK FIXTURE — do not modify
# Contains intentional bugs. Test functions at bottom verify fixes.


def safe_divide(a, b):
    # BUG 1: no division by zero check
    return a / b


def get_last_item(items):
    # BUG 2: off-by-one — should be len(items) - 1
    return items[len(items)]


def is_adult(age):
    # BUG 3: wrong operator — should be >= not >
    if age > 18:
        return True
    return False


def get_full_name(first, last):
    # BUG 4: missing return statement
    full = f"{first} {last}"


# ── Verifier tests — these must pass after fixes ──────────────────────────────

def test_safe_divide_zero():
    try:
        result = safe_divide(10, 0)
        # Should not reach here — must handle division by zero
        assert False, "Should have handled division by zero"
    except (ZeroDivisionError, ValueError):
        pass  # Either raise or return a safe value is acceptable
    # Also verify normal division still works
    assert safe_divide(10, 2) == 5.0

def test_get_last_item():
    items = [1, 2, 3, 4, 5]
    assert get_last_item(items) == 5, f"Expected 5, got {get_last_item(items)}"

def test_is_adult():
    assert is_adult(18) == True, "18 year old should be adult"
    assert is_adult(17) == False, "17 year old should not be adult"
    assert is_adult(21) == True, "21 year old should be adult"

def test_get_full_name():
    result = get_full_name("John", "Doe")
    assert result == "John Doe", f"Expected 'John Doe', got {result!r}"

# Run all tests
ALL_TESTS = [
    test_safe_divide_zero,
    test_get_last_item,
    test_is_adult,
    test_get_full_name,
]
