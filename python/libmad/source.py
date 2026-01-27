from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _require_tool(name: str, hint: str | None = None) -> str:
    path = shutil.which(name)
    if path:
        return path
    message = f"Required tool not found: {name}"
    if hint:
        message = f"{message}. {hint}"
    raise RuntimeError(message)


def _find_built_lib(build_dir: Path) -> Path | None:
    candidates = [
        build_dir / "lib" / "libMad.so",
        build_dir / "lib" / "libMad.dylib",
        build_dir / "lib" / "libMad.dll",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_from_source(
    repo_path: str | os.PathLike,
    build_dir: str | os.PathLike | None = None,
    clean: bool = True,
    extra_cmake_args: list[str] | None = None,
) -> dict:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"libMad repo not found: {repo}")
    if not (repo / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"CMakeLists.txt not found in repo: {repo}")

    _require_tool("julia", "Install Julia 1.12+ with JuliaC app and ensure both are on PATH.")
    _require_tool("cmake", "Install CMake or `python -m pip install cmake`.")

    build = Path(build_dir).expanduser().resolve() if build_dir else (repo / "build")
    if clean and build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)

    cmake_args = ["cmake", str(repo)]
    juliac_env = os.environ.get("JULIAC_EXECUTABLE")
    if juliac_env:
        cmake_args.append(f"-DJULIAC_EXECUTABLE={juliac_env}")
    if extra_cmake_args:
        cmake_args.extend(extra_cmake_args)

    subprocess.run(cmake_args, check=True, cwd=build)
    subprocess.run(["cmake", "--build", str(build)], check=True)

    lib_path = _find_built_lib(build)
    header_path = build / "include" / "libMad.h"

    return {
        "build_dir": str(build),
        "lib_path": str(lib_path) if lib_path else None,
        "header_path": str(header_path) if header_path.is_file() else None,
    }


def build_from_github(
    repo_url: str,
    branch: str,
    build_dir: str | os.PathLike | None = None,
    clean: bool = True,
    extra_cmake_args: list[str] | None = None,
) -> dict:
    _require_tool("git", "Install git and ensure it is on PATH.")

    tmp_root = Path(tempfile.mkdtemp(prefix="libmad-src-"))
    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    if not repo_name:
        repo_name = "libMad"
    repo_dir = tmp_root / repo_name
    try:
        subprocess.run(
            ["git", "clone", "--recursive", repo_url, str(repo_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", branch],
            check=True,
            cwd=repo_dir,
        )
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            check=True,
            cwd=repo_dir,
        )
        return build_from_source(
            repo_dir,
            build_dir=build_dir,
            clean=clean,
            extra_cmake_args=extra_cmake_args,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
