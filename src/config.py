"""
Centralized configuration for paths and environment variables.
Ensures compatibility across macOS (local dev) and Ubuntu (GitHub Actions).

All paths are repo-relative unless explicitly overridden by environment variables.
"""

import os
from pathlib import Path


def get_workspace_root() -> Path:
    """Get the repository root, respecting GitHub Actions workspace."""
    return Path(os.getenv("GITHUB_WORKSPACE", Path.cwd())).resolve()


def safe_dir(*parts: str) -> Path:
    """Create and return a safe, repo-relative directory path.
    
    Args:
        *parts: Path components relative to workspace root
        
    Returns:
        Path object for the directory, created if it doesn't exist
    """
    root = get_workspace_root()
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


class PathConfig:
    """Centralized path configuration."""

    _workspace_root = get_workspace_root()

    @classmethod
    def workspace_root(cls) -> Path:
        """Repository root directory."""
        return cls._workspace_root

    @classmethod
    def data_raw(cls) -> Path:
        """Raw data directory."""
        env_path = os.getenv("DATA_RAW_DIR")
        if env_path:
            path = Path(env_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return safe_dir("data", "raw")

    @classmethod
    def data_processed(cls) -> Path:
        """Processed data directory."""
        env_path = os.getenv("DATA_PROCESSED_DIR")
        if env_path:
            path = Path(env_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return safe_dir("data", "processed")

    @classmethod
    def models(cls) -> Path:
        """Model artifacts directory."""
        env_path = os.getenv("MODEL_DIR")
        if env_path:
            path = Path(env_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return safe_dir("models")

    @classmethod
    def evaluation_output(cls) -> Path:
        """Evaluation output directory."""
        env_path = os.getenv("EVALUATION_OUTPUT_DIR")
        if env_path:
            path = Path(env_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return safe_dir("evaluation_output")

    @classmethod
    def mlflow_tracking(cls) -> str:
        """MLflow tracking URI (local directory)."""
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri
        tracking_dir = safe_dir("mlruns")
        return str(tracking_dir)

    @classmethod
    def artifacts(cls) -> Path:
        """Artifacts directory for caching, temp files, etc."""
        env_path = os.getenv("ARTIFACTS_DIR")
        if env_path:
            path = Path(env_path).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return safe_dir("artifacts")


def ensure_safe_environment() -> None:
    """Set safe environment variables for GitHub Actions / non-macOS systems.
    
    This prevents hardcoded /Users/... paths from being created or accessed.
    Should be called once at pipeline start.
    """
    workspace = get_workspace_root()
    
    # Force HOME to workspace (override any system HOME that may contain /Users)
    # This is critical for tools like kagglehub, sklearn cache, etc.
    os.environ["HOME"] = str(workspace)
    
    # Force cache directories in workspace
    os.environ["XDG_CACHE_HOME"] = str(workspace / ".cache")
    os.environ["KAGGLEHUB_CACHE"] = str(workspace / ".cache" / "kagglehub")
    os.environ["TMPDIR"] = str(workspace / ".tmp")
    
    # Ensure sklearn cache is also in workspace
    os.environ["SKLEARN_CONFIG_HOME"] = str(workspace / ".cache" / "sklearn")
    
    # Ensure MLflow doesn't create paths outside workspace
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = str(safe_dir("mlruns"))
