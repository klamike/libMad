import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir


_DEFAULT_GITHUB_REPO = os.environ.get("LIBMAD_GITHUB_REPO", "apozharski/libMad")
_DEFAULT_RELEASE = os.environ.get("LIBMAD_RELEASE", "latest")
_DEFAULT_RELEASES_BASE = os.environ.get("LIBMAD_RELEASES_BASE", f"https://github.com/{_DEFAULT_GITHUB_REPO}/releases/download")


def _lib_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libMad.dylib"
    if system == "Linux":
        return "libMad.so"
    if system == "Windows":
        return "libMad.dll"
    raise RuntimeError(f"Unsupported platform: {system}")


def _lib_subdir() -> str:
    return "bin" if platform.system() == "Windows" else "lib"


def _platform_asset_key() -> str:
    override = os.environ.get("LIBMAD_PLATFORM")
    if override:
        return override

    system = platform.system()
    machine = platform.machine().lower()

    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "ubuntu-x86_64"
        if machine in ("aarch64", "arm64"):
            return "ubuntu-aarch64"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "apple-aarch64"
        if machine in ("x86_64", "amd64"):
            return "apple-x86_64"
    elif system == "Windows":
        if machine in ("x86_64", "amd64"):
            return "windows-x86_64"

    raise RuntimeError(f"Unsupported platform/arch: {system}/{machine}")


def default_cache_dir() -> str:
    override = os.environ.get("LIBMAD_CACHE_DIR")
    if override:
        return override

    return user_cache_dir("libmad")


def _github_latest_release(repo: str) -> tuple[str, list[dict]]:
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["tag_name"], data.get("assets", [])


def _select_asset_url(tag: str, assets: list[dict], platform_key: str) -> str:
    override = os.environ.get("LIBMAD_ASSET_URL")
    if override:
        return override

    if assets:
        expected = f"libMad-{platform_key}-{tag}.tar.gz"
        for asset in assets:
            if asset.get("name") == expected:
                return asset.get("browser_download_url")

    return f"{_DEFAULT_RELEASES_BASE}/{tag}/libMad-{platform_key}-{tag}.tar.gz"


def _download_file(url: str, dst_path) -> None:
    headers = {"User-Agent": "libmad-python"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dst_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def resolve_lib_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_dir():
        candidate = path_obj / _lib_filename()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"libMad not found in directory: {path}")
    if path_obj.is_file():
        return str(path_obj)
    raise FileNotFoundError(f"LIBMAD_PATH does not exist: {path}")

def default_download() -> str:
    cache_dir = default_cache_dir()
    cached = find_cached_lib(cache_dir)
    if cached:
        return cached
    return download_and_extract(_DEFAULT_RELEASE, _platform_asset_key(), cache_dir)

def download_and_extract(tag: str, platform_key: str, cache_dir: str) -> str:
    if os.environ.get("LIBMAD_NO_DOWNLOAD"):
        raise RuntimeError("libMad download disabled (LIBMAD_NO_DOWNLOAD=1).")

    assets = []
    if _DEFAULT_RELEASE == "latest":
        tag, assets = _github_latest_release(_DEFAULT_GITHUB_REPO)

    url = _select_asset_url(tag, assets, platform_key)
    cache_path = Path(cache_dir)
    install_dir = cache_path / tag / platform_key
    lib_dir = install_dir / _lib_subdir()
    lib_path = lib_dir / _lib_filename()

    if lib_path.is_file():
        return str(lib_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="libmad-", dir=str(cache_path)))
    try:
        archive_path = tmp_root / "libmad.tar.gz"
        _download_file(url, archive_path)

        extract_dir = tmp_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        if install_dir.is_dir():
            shutil.rmtree(install_dir)
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extract_dir), str(install_dir))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not lib_path.is_file():
        raise FileNotFoundError(
            f"libMad library not found after download: {lib_path}"
        )
    return str(lib_path)


def find_cached_lib(cache_dir: str) -> str | None:
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return None

    platform_key = _platform_asset_key()
    lib_name = _lib_filename()

    if _DEFAULT_RELEASE != "latest":
        candidate = cache_path / _DEFAULT_RELEASE / platform_key / _lib_subdir() / lib_name
        return str(candidate) if candidate.is_file() else None

    matches = list(cache_path.glob(f"*/{platform_key}/{_lib_subdir()}/{lib_name}"))
    if not matches:
        return None

    newest = max(matches, key=lambda p: p.stat().st_mtime)
    return str(newest)
