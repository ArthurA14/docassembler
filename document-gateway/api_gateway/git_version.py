import subprocess


def current_git_version():
    """Return the current git tag version, falling back to '0.0.0' on failure."""
    try:
        return (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "0.0.0"