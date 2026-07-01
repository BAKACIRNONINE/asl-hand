r"""
git_auto_create_push_v4.py

Fixes:
- if origin points to missing GitHub repo, remove it and create repo
- verifies repo exists before push
- creates repo from folder name, asks private/public
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
import webbrowser
from datetime import datetime
from pathlib import Path


VERSION = "v4-fix-missing-origin-repo"

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


class Progress:
    def __init__(self, total=8):
        self.total = total
        self.current = 0

    def step(self, text):
        self.current += 1
        width = 22
        filled = int(width * self.current / self.total)
        print(f"\n[{self.current:02d}/{self.total:02d}] {'█' * filled}{'░' * (width - filled)} {text}")
        time.sleep(0.03)


def run(cmd, cwd=None, check=True):
    print("$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r


def cap(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def find_git():
    found = shutil.which("git")
    if found:
        return found
    for p in [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    ]:
        if Path(p).exists():
            return p
    raise SystemExit("Git not found")


def find_gh():
    found = shutil.which("gh")
    if found:
        return found
    for p in [
        r"C:\Program Files\GitHub CLI\gh.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe"),
    ]:
        if Path(p).exists():
            return p
    raise SystemExit("GitHub CLI not found")


def slugify(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-._")
    return name or "project-backup"


def get_remote(git, project):
    code, out, _ = cap([git, "-C", str(project), "remote", "get-url", "origin"])
    return out if code == 0 and out else None


def remove_origin(git, project):
    if get_remote(git, project):
        run([git, "-C", str(project), "remote", "remove", "origin"], check=False)


def remote_to_web(remote):
    if remote.startswith("https://github.com/"):
        return remote[:-4] if remote.endswith(".git") else remote
    if remote.startswith("git@github.com:"):
        u = remote.replace("git@github.com:", "https://github.com/")
        return u[:-4] if u.endswith(".git") else u
    return None


def repo_full_name_from_remote(remote):
    web = remote_to_web(remote)
    if not web:
        return None
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)$", web)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def gh_repo_exists(gh, full_name):
    if not full_name:
        return False
    code, _, _ = cap([gh, "repo", "view", full_name])
    return code == 0


def open_repo(git, project):
    remote = get_remote(git, project)
    if not remote:
        print("[WARN] No origin remote.")
        return
    web = remote_to_web(remote)
    if web:
        print(f"[OPEN] {web}")
        webbrowser.open(web)


def ensure_files(project):
    gi = project / ".gitignore"
    text = gi.read_text(encoding="utf-8", errors="ignore") if gi.exists() else ""
    if "AI_PROJECT_BACKUP_RULES_BEGIN" not in text:
        with gi.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n"):
                f.write("\n")
            f.write("\n" + IGNORE_BLOCK + "\n")

    readme = project / "README.md"
    if not readme.exists():
        readme.write_text(f"# {project.name}\n\nCode and guidance files are stored here.\n", encoding="utf-8")


def ensure_git_repo(git, project):
    run([git, "-C", str(project), "init"])
    run([git, "-C", str(project), "checkout", "-B", "main"], check=False)
    return "main"


def commit(git, project, msg=None):
    run([git, "-C", str(project), "status", "--short"], check=False)
    run([git, "-C", str(project), "add", "."])
    code, _, _ = cap([git, "-C", str(project), "diff", "--cached", "--quiet"])
    if code == 0:
        print("[OK] Nothing to commit.")
        return

    msg = msg or f"Backup project {datetime.now().strftime('%Y-%m-%d %H-%M')}"
    r = subprocess.run([git, "-C", str(project), "commit", "-m", msg], text=True)
    if r.returncode != 0:
        print("[WARN] Git identity missing.")
        name = input("Git user.name: ").strip()
        email = input("Git user.email: ").strip()
        if name:
            run([git, "config", "--global", "user.name", name])
        if email:
            run([git, "config", "--global", "user.email", email])
        run([git, "-C", str(project), "commit", "-m", msg])


def choose_visibility(args):
    if args.public:
        return True
    if args.private:
        return False
    print("\nChoose GitHub repo visibility:")
    print("1. private")
    print("2. public")
    while True:
        c = input("Enter 1/2, Enter=1: ").strip() or "1"
        if c == "1":
            return False
        if c == "2":
            return True


def gh_login(gh):
    if subprocess.run([gh, "auth", "status"], text=True).returncode != 0:
        run([gh, "auth", "login"])


def gh_username(gh):
    code, out, _ = cap([gh, "api", "user", "-q", ".login"])
    return out if code == 0 and out else None


def ensure_valid_origin(git, gh, project, args):
    gh_login(gh)

    remote = get_remote(git, project)
    if remote:
        full_name = repo_full_name_from_remote(remote)
        if gh_repo_exists(gh, full_name):
            print(f"[ORIGIN OK] {remote}")
            return
        print(f"[WARN] origin points to missing repo: {remote}")
        remove_origin(git, project)

    repo = slugify(project.name)
    public = choose_visibility(args)
    visibility = "--public" if public else "--private"

    print(f"[INFO] Repo name: {repo}")
    print(f"[INFO] Visibility: {'public' if public else 'private'}")

    # If repo already exists on account, use it.
    user = gh_username(gh)
    if user and gh_repo_exists(gh, f"{user}/{repo}"):
        url = f"https://github.com/{user}/{repo}.git"
        print(f"[INFO] Existing repo found: {url}")
        run([git, "-C", str(project), "remote", "add", "origin", url])
        return

    r = subprocess.run(
        [gh, "repo", "create", repo, visibility, "--source", str(project), "--remote", "origin"],
        cwd=project,
        text=True,
    )

    if r.returncode != 0:
        print("[WARN] Create failed. Enter a different repo name.")
        new_repo = input(f"Repo name, Enter for {repo}-2: ").strip() or f"{repo}-2"
        run([gh, "repo", "create", new_repo, visibility, "--source", str(project), "--remote", "origin"], cwd=project)


def push(git, project):
    r = subprocess.run([git, "-C", str(project), "push", "-u", "origin", "main"], text=True)
    if r.returncode == 0:
        return

    print("[WARN] Push failed. Checking origin...")
    raise SystemExit(r.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument("--message", "-m", default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    print("=" * 80)
    print(f"git_auto_create_push_v4.py | {VERSION}")
    print(f"[PROJECT] {project}")
    print("=" * 80)

    p = Progress()

    p.step("Check Git")
    git = find_git()
    run([git, "--version"], check=False)

    if args.open_only:
        p.step("Open repo")
        open_repo(git, project)
        return

    p.step("Check GitHub CLI")
    gh = find_gh()

    p.step("Prepare files")
    ensure_files(project)

    p.step("Force git init")
    ensure_git_repo(git, project)

    p.step("Commit changes")
    commit(git, project, args.message)

    p.step("Create/link valid GitHub origin")
    ensure_valid_origin(git, gh, project, args)

    p.step("Push")
    push(git, project)

    p.step("Open repo")
    open_repo(git, project)

    print("[DONE]")


if __name__ == "__main__":
    main()
