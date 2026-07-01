r"""
git_auto_link_push.py

One-click GitHub link + commit + push helper for Windows / VS Code projects.

Usage:
  python ai_guidance/git_auto_link_push.py --project "C:\Users\haha2\Desktop\CS\ASL Hand"
  python ai_guidance/git_auto_link_push.py --remote https://github.com/USER/REPO.git
  python ai_guidance/git_auto_link_push.py --create-repo --repo-name asl-hand-recognition --private
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path


IGNORE_BLOCK = """
# AI_PROJECT_BACKUP_RULES_BEGIN
__pycache__/
*.pyc
.venv/
venv/
env/
.env
*.env
kaggle.json
.DS_Store
Thumbs.db

datasets/
data/
runs/
outputs/
weights/
checkpoints/

*.pth
*.pt
*.onnx
*.engine
*.joblib
*.pkl

*.mp4
*.mov
*.avi
*.mkv
*.zip
*.rar
*.7z
# AI_PROJECT_BACKUP_RULES_END
""".strip()


def run(cmd, cwd: Path | None = None, check: bool = True):
    print("\n$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def capture(cmd, cwd: Path | None = None):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def find_git() -> str | None:
    found = shutil.which("git")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    ]

    for item in candidates:
        if Path(item).exists():
            return item

    return None


def find_gh() -> str | None:
    found = shutil.which("gh")
    if found:
        return found

    candidates = [
        r"C:\Program Files\GitHub CLI\gh.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe"),
    ]

    for item in candidates:
        if Path(item).exists():
            return item

    return None


def install_with_winget(package_id: str):
    if not shutil.which("winget"):
        raise SystemExit("winget not found. Please install manually.")

    cmd = [
        "winget", "install",
        "--id", package_id,
        "-e",
        "--source", "winget",
        "--scope", "machine",
        "--accept-source-agreements",
        "--accept-package-agreements",
    ]

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        cmd.remove("--scope")
        cmd.remove("machine")
        run(cmd)


def ensure_git(install_git: bool) -> str:
    git = find_git()
    if git:
        return git

    if not install_git:
        raise SystemExit("Git not found. Re-run with --install-git or install Git for Windows.")

    print("[INFO] Git not found. Installing Git with winget...")
    install_with_winget("Git.Git")

    git = find_git()
    if not git:
        raise SystemExit("Git installed but not found. Reopen VS Code/terminal and run again.")

    return git


def ensure_gitignore(project: Path):
    path = project / ".gitignore"
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

    if "AI_PROJECT_BACKUP_RULES_BEGIN" not in text:
        with path.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n"):
                f.write("\n")
            f.write("\n" + IGNORE_BLOCK + "\n")

    print("[OK] .gitignore ready")


def ensure_readme(project: Path):
    path = project / "README.md"
    if path.exists():
        return

    path.write_text(
        f"# {project.name}\n\n"
        "Project code and guidance files are stored here.\n\n"
        "Large datasets, model weights, generated runs, and media files are ignored by default.\n",
        encoding="utf-8",
    )
    print("[OK] README.md created")


def ensure_vscode_helpers(project: Path):
    vscode = project / ".vscode"
    vscode.mkdir(exist_ok=True)

    settings_path = vscode / "settings.json"
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            settings = {}

    settings.update({
        "git.autofetch": True,
        "git.confirmSync": False,
        "files.autoSave": "afterDelay",
    })
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks_path = vscode / "tasks.json"
    if not tasks_path.exists():
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Git: Commit and Push",
                    "type": "shell",
                    "command": "python ai_guidance/git_auto_link_push.py",
                    "problemMatcher": [],
                    "group": "build"
                }
            ]
        }
        tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] VS Code helpers ready")


def get_remote(git: str, project: Path) -> str | None:
    code, out, _ = capture([git, "remote", "get-url", "origin"], cwd=project)
    return out if code == 0 and out else None


def remote_to_web(remote: str) -> str | None:
    if remote.startswith("https://github.com/"):
        return remote[:-4] if remote.endswith(".git") else remote

    if remote.startswith("git@github.com:"):
        url = remote.replace("git@github.com:", "https://github.com/")
        return url[:-4] if url.endswith(".git") else url

    if remote.startswith("ssh://git@github.com/"):
        url = remote.replace("ssh://git@github.com/", "https://github.com/")
        return url[:-4] if url.endswith(".git") else url

    return None


def ensure_remote(git: str, project: Path, args):
    remote = get_remote(git, project)
    if remote:
        print(f"[ORIGIN] {remote}")
        return

    if args.remote:
        run([git, "remote", "add", "origin", args.remote], cwd=project)
        return

    if args.create_repo:
        gh = find_gh()
        if not gh:
            if args.install_gh:
                print("[INFO] Installing GitHub CLI with winget...")
                install_with_winget("GitHub.cli")
                gh = find_gh()
            else:
                raise SystemExit("GitHub CLI not found. Use --install-gh or --remote URL.")

        if subprocess.run([gh, "auth", "status"], text=True).returncode != 0:
            run([gh, "auth", "login"])

        repo_name = args.repo_name or project.name.replace(" ", "-")
        visibility = "--public" if args.public else "--private"
        run([gh, "repo", "create", repo_name, visibility, "--source=.", "--remote=origin"], cwd=project)
        return

    remote = input("No origin. Paste GitHub repo URL, or press Enter to skip: ").strip()
    if remote:
        run([git, "remote", "add", "origin", remote], cwd=project)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--remote", type=str, default=None)
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--repo-name", type=str, default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", "-m", type=str, default=None)
    parser.add_argument("--install-git", action="store_true")
    parser.add_argument("--install-gh", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    if not project.exists():
        raise SystemExit(f"Project folder not found: {project}")

    print(f"[PROJECT] {project}")

    git = ensure_git(args.install_git)
    run([git, "--version"], check=False)

    ensure_gitignore(project)
    ensure_readme(project)
    ensure_vscode_helpers(project)

    code, _, _ = capture([git, "rev-parse", "--is-inside-work-tree"], cwd=project)
    if code != 0:
        run([git, "init"], cwd=project)
        run([git, "branch", "-M", "main"], cwd=project)

    code, branch, _ = capture([git, "branch", "--show-current"], cwd=project)
    if not branch:
        branch = "main"
        run([git, "checkout", "-B", branch], cwd=project)

    ensure_remote(git, project, args)

    run([git, "status", "--short"], cwd=project, check=False)
    run([git, "add", "."], cwd=project)

    code, _, _ = capture([git, "diff", "--cached", "--quiet"], cwd=project)
    if code != 0:
        message = args.message or f"Backup project {datetime.now().strftime('%Y-%m-%d %H-%M')}"
        result = subprocess.run([git, "commit", "-m", message], cwd=project, text=True)
        if result.returncode != 0:
            name = input("Git user.name: ").strip()
            email = input("Git user.email: ").strip()
            if name:
                run([git, "config", "--global", "user.name", name])
            if email:
                run([git, "config", "--global", "user.email", email])
            run([git, "commit", "-m", message], cwd=project)
    else:
        print("[OK] Nothing to commit.")

    remote = get_remote(git, project)

    if remote and not args.no_push:
        push = input("Push to GitHub now? 1=yes 2=no: ").strip()
        if push == "1":
            run([git, "push", "-u", "origin", branch], cwd=project)
            print("[OK] Uploaded.")
            web = remote_to_web(get_remote(git, project) or remote)
            if web:
                print(f"[OPEN] {web}")
                webbrowser.open(web)

    print("[DONE]")


if __name__ == "__main__":
    main()
