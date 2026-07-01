r"""
git_auto_push_simple_v2.py

功能：
- 自动检查 Git / GitHub CLI
- 自动检查 GitHub 是否已登录
- 已登录：不会重复登录，直接 push
- 未登录：只登录一次，并自动执行 gh auth setup-git
- 自动 git init
- 自动创建 .gitignore
- 自动 commit
- 没有 origin 或 origin 仓库不存在：按文件夹名自动创建 GitHub 仓库
- 可选 public/private
- 自动 push
- 自动打开 GitHub 仓库网页

运行：
  python "ai_guidance/git_auto_push_simple_v2.py"

指定公开：
  python "ai_guidance/git_auto_push_simple_v2.py" --public

指定私有：
  python "ai_guidance/git_auto_push_simple_v2.py" --private
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path


VERSION = "simple-v2-login-cache-auto-create-push"

IGNORE_BLOCK = """
# AI_BACKUP_IGNORE_BEGIN
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
# AI_BACKUP_IGNORE_END
""".strip()


def cmd(args, cwd=None, check=True, capture=False):
    print("$ " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in args))
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r


def output(args, cwd=None):
    r = cmd(args, cwd=cwd, check=False, capture=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def find_exe(name, candidates):
    found = shutil.which(name)
    if found:
        return found

    for p in candidates:
        if Path(p).exists():
            return p

    raise SystemExit(f"[ERROR] {name} not found")


def slug(name):
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-._")
    return name or "project-backup"


def add_ignore(project):
    p = project / ".gitignore"
    text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""

    if "AI_BACKUP_IGNORE_BEGIN" not in text:
        p.write_text((text.rstrip() + "\n\n" + IGNORE_BLOCK + "\n").lstrip(), encoding="utf-8")


def ensure_git_repo(git, project):
    cmd([git, "-C", str(project), "init"])
    cmd([git, "-C", str(project), "checkout", "-B", "main"], check=False)


def ensure_git_identity(git):
    code_name, name, _ = output([git, "config", "--global", "user.name"])
    code_email, email, _ = output([git, "config", "--global", "user.email"])

    if code_name == 0 and name and code_email == 0 and email:
        return

    print("[WARN] Git commit identity is missing.")
    new_name = input("Git user.name: ").strip()
    new_email = input("Git user.email: ").strip()

    if new_name:
        cmd([git, "config", "--global", "user.name", new_name])
    if new_email:
        cmd([git, "config", "--global", "user.email", new_email])


def gh_logged_in(gh):
    code, _, _ = output([gh, "auth", "status", "-h", "github.com"])
    return code == 0


def ensure_github_login(gh, git):
    if gh_logged_in(gh):
        print("[OK] GitHub CLI already logged in")
    else:
        print("[LOGIN] GitHub CLI not logged in. Browser login is required once.")
        cmd([gh, "auth", "login", "-h", "github.com", "-p", "https", "-w"])

    # Make normal git push reuse GitHub CLI credentials.
    cmd([gh, "auth", "setup-git", "-h", "github.com"], check=False)

    # Make Git Credential Manager remember credentials.
    cmd([git, "config", "--global", "credential.helper", "manager"], check=False)


def github_user(gh):
    code, out, _ = output([gh, "api", "user", "-q", ".login"])
    if code != 0 or not out:
        raise SystemExit("[ERROR] Cannot read GitHub username. Please run: gh auth login")
    return out


def get_origin(git, project):
    code, out, _ = output([git, "-C", str(project), "remote", "get-url", "origin"])
    return out if code == 0 and out else None


def remove_origin(git, project):
    if get_origin(git, project):
        cmd([git, "-C", str(project), "remote", "remove", "origin"], check=False)


def parse_full_name(origin):
    if not origin:
        return None

    origin = origin.strip()
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?/?$", origin)

    if not m:
        return None

    return f"{m.group(1)}/{m.group(2)}"


def repo_exists(gh, full_name):
    if not full_name:
        return False
    code, _, _ = output([gh, "repo", "view", full_name])
    return code == 0


def choose_visibility(args):
    if args.public:
        return True
    if args.private:
        return False

    print("\n选择 GitHub 仓库权限：")
    print("1. private 私有")
    print("2. public 公开")

    while True:
        value = input("输入 1/2，回车默认 1: ").strip() or "1"
        if value == "1":
            return False
        if value == "2":
            return True
        print("[ERROR] 请输入 1 或 2")


def ensure_origin(git, gh, project, args):
    origin = get_origin(git, project)

    if origin:
        full_name = parse_full_name(origin)
        if repo_exists(gh, full_name):
            print(f"[OK] origin exists: {origin}")
            return origin

        print(f"[WARN] origin points to missing repo, removing: {origin}")
        remove_origin(git, project)
        origin = None

    user = github_user(gh)
    repo = slug(project.name)
    full_name = f"{user}/{repo}"

    if not repo_exists(gh, full_name):
        is_public = choose_visibility(args)
        visibility = "--public" if is_public else "--private"
        print(f"[CREATE] {full_name} ({'public' if is_public else 'private'})")
        cmd([gh, "repo", "create", repo, visibility])
    else:
        print(f"[OK] GitHub repo already exists: {full_name}")

    origin = f"https://github.com/{full_name}.git"
    cmd([git, "-C", str(project), "remote", "add", "origin", origin], check=False)
    cmd([git, "-C", str(project), "remote", "set-url", "origin", origin], check=False)
    return origin


def commit_all(git, project, message):
    cmd([git, "-C", str(project), "status", "--short"], check=False)
    cmd([git, "-C", str(project), "add", "."])

    code, _, _ = output([git, "-C", str(project), "diff", "--cached", "--quiet"])
    if code == 0:
        print("[OK] No new changes. Skip commit.")
        return

    msg = message or f"Backup {datetime.now().strftime('%Y-%m-%d %H-%M')}"
    r = cmd([git, "-C", str(project), "commit", "-m", msg], check=False)

    if r.returncode != 0:
        ensure_git_identity(git)
        cmd([git, "-C", str(project), "commit", "-m", msg])


def open_repo(origin):
    url = origin
    if url.endswith(".git"):
        url = url[:-4]
    url = url.replace("git@github.com:", "https://github.com/")
    print(f"[OPEN] {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    project = args.project.resolve()

    print("=" * 80)
    print(f"git_auto_push_simple_v2.py | {VERSION}")
    print(f"[PROJECT] {project}")
    print("=" * 80)

    git = find_exe("git", [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ])
    gh = find_exe("gh", [
        r"C:\Program Files\GitHub CLI\gh.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe"),
    ])

    ensure_git_repo(git, project)

    if args.open_only:
        origin = get_origin(git, project)
        if origin:
            open_repo(origin)
        else:
            print("[WARN] No origin")
        return

    add_ignore(project)
    ensure_github_login(gh, git)

    origin = ensure_origin(git, gh, project, args)
    commit_all(git, project, args.message)

    cmd([git, "-C", str(project), "push", "-u", "origin", "main"])
    open_repo(origin)

    print("[DONE]")


if __name__ == "__main__":
    main()
