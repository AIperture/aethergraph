# aethergraph/services/execution/local_python.py
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from aethergraph.contracts.services.execution import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ExecutionService,
)


class LocalPythonExecutionService(ExecutionService):
    """
    Minimal execution service that runs Python code using the local interpreter.

    This is NOT sandboxed. It runs with the same permissions as the AG process.
    Use only for experimentation or very trusted code.
    """

    def __init__(self, python_executable: str = "python"):
        self.python_executable = python_executable

    async def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if request.language != "python":
            return CodeExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error=f"Unsupported language: {request.language}",
                metadata={"reason": "unsupported_language"},
            )

        # Write the code to a temporary .py file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "script.py"
            script_path.write_text(request.code, encoding="utf-8")

            cmd: list[str] = [self.python_executable, str(script_path)]
            if request.args:
                cmd.extend(request.args)

            # Working directory and environment (optional, for future)
            cwd = request.workdir or tmpdir
            env: dict[str, str] | None = None
            if request.env is not None:
                # Start from current env, update with overrides
                import os

                env = os.environ.copy()
                env.update(request.env)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=request.timeout_s
                    )
                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")

                    return CodeExecutionResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=proc.returncode,
                        error=None,
                        metadata={"timeout_s": request.timeout_s},
                    )
                except asyncio.TimeoutError:
                    # Kill the process on timeout
                    proc.kill()
                    await proc.wait()
                    return CodeExecutionResult(
                        stdout="",
                        stderr="Execution timed out",
                        exit_code=-1,
                        error="timeout",
                        metadata={"timeout_s": request.timeout_s},
                    )

            except Exception as exc:
                return CodeExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=-1,
                    error="spawn_failed",
                    metadata={"exception_type": type(exc).__name__},
                )
