# System Patterns *Optional*

This file documents recurring patterns and standards used in the project.
It is optional, but recommended to be updated as the project evolves.
2025-06-01 10:42:02 - Log of updates made.

*

## Coding Patterns

*   

## Architectural Patterns

*   

## Testing Patterns

*
---
### Testing Patterns
[2025-06-01 10:46:02] - PySP 项目测试模式基于 `pytest` 框架。

**1. Fixture-Based Setup and Teardown:**
   *   **Purpose:** Manage test dependencies, ensure clean state for each test, and promote reusability.
   *   **Implementation:**
        *   Define fixtures in `test/conftest.py` for project-wide shared resources (e.g., `base_sample_rate`, `temp_wav_file`).
        *   Define module-specific fixtures directly within `test_*.py` files if they are not shared.
        *   Use fixture scopes (`function`, `class`, `module`, `session`) appropriately to optimize setup/teardown overhead.
        *   Example: `short_sine_wave_signal` fixture in `conftest.py` provides a consistent `Signal` object for many tests.

**2. Parameterized Testing for Variety:**
   *   **Purpose:** Test functions/methods with multiple input combinations without duplicating test code.
   *   **Implementation:** Use `@pytest.mark.parametrize("arg1, arg2, expected", [(val1_a, val2_a, exp_a), (val1_b, val2_b, exp_b)])`.
   *   Example: Testing `Signal.slice()` with different `start_time` and `end_time` values.

**3. Mocking for Isolation:**
   *   **Purpose:** Isolate unit tests from external dependencies (e.g., file system, network, complex libraries like `matplotlib`).
   *   **Implementation:** Use `pytest-mock` (via the `mocker` fixture) or `unittest.mock.patch`.
   *   Example: In `test_Plot.py`, `mocker.patch('PySP.Plot.plt')` is used to mock `matplotlib.pyplot` calls and assert their arguments.

**4. Clear and Specific Assertions:**
   *   **Purpose:** Ensure test failures provide immediate and understandable feedback.
   *   **Implementation:** Use `pytest`'s native assertions. For floating-point comparisons, use `pytest.approx()`. For array comparisons, use `numpy.array_equal()` or `numpy.allclose()`.
   *   Example: `assert sig.duration == pytest.approx(expected_duration)`.

**5. Test Naming Conventions:**
   *   Files: `test_*.py` (e.g., `test_Signal.py`)
   *   Classes: `Test*` (e.g., `TestSignalInitialization`)
   *   Functions/Methods: `test_*` (e.g., `test_save_and_load_wav`)
   *   Names should be descriptive of the behavior being tested.

**6. Test Coverage Monitoring:**
   *   **Purpose:** Track how much of the codebase is exercised by tests.
   *   **Implementation:** Utilize the `pytest-cov` plugin. Aim for high coverage of critical modules and logic.
   *   Command: `pytest --cov=PySP` (or similar, depending on project structure).

**7. Handling Test Data:**
   *   Small, inline data: Define directly in tests or fixtures.
   *   Larger or shared data: Consider placing in a `test/test_data/` directory and loading within fixtures (e.g., reference WAV files, expected analysis outputs as CSV/JSON).