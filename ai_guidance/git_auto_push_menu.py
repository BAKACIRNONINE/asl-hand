r"""
git_auto_push_menu.py

带菜单版：
1. 智能自动：有可用 origin 就 push；没有就创建仓库再 push
2. 只 push：不创建仓库
3. 创建/重连仓库：按文件夹名创建 GitHub 仓库，再 push
4. 打开当前 GitHub 仓库页面
0. 退出

运行：
  python "ai_guidance/git_auto_push_menu.py"
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


VERSION = "menu-v1-smart-push-create"

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


def out(args, cwd=None):
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


def menu():
    print("\n选择操作：")
    print("1. 智能自动：有 origin 就 push；没有就创建仓库再 push")
    print("2. 只 push 到已有 origin，不创建仓库")
    print("3. 创建/重连 GitHub 仓库，然后 push")
    print("4. 打开当前 GitHub 仓库页面")
    print("0. 退出")

    while True:
        c = input("输入 0/1/2/3/4，回车默认 1: ").strip() or "1"
        if c in {"0", "1", "2", "3", "4"}:
            return c
        print("[ERROR] 请输入 0/1/2/3/4")


def choose_visibility(args):
    if args.public:
        return True
    if args.private:
        return False

    print("\n选择仓库权限：")
    print("1. private 私有")
    print("2. public 公开")

    while True:
        c = input("输入 1/2，回车默认 1: ").strip() or "1"
        if c == "1":
            return False
        if c == "2":
            return True
        print("[ERROR] 请输入 1 或 2")


def add_ignore(project):
    p = project / ".gitignore"
    text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if "AI_BACKUP_IGNORE_BEGIN" not in text:
        p.write_text((text.rstrip() + "\n\n" + IGNORE_BLOCK + "\n").lstrip(), encoding="utf-8")


def ensure_git_repo(git, project):
    cmd([git, "-C", str(project), "init"])
    cmd([git, "-C", str(project), "checkout", "-B", "main"], check=False)


def ensure_github_login(gh, git):
    code, _, _ = out([gh, "auth", "status", "-h", "github.com"])
    if code == 0:
        print("[OK] GitHub 已登录，不重复登录")
    else:
        print("[LOGIN] GitHub 未登录，需要浏览器登录一次")
        cmd([gh, "auth", "login", "-h", "github.com", "-p", "https", "-w"])

    cmd([gh, "auth", "setup-git", "-h", "github.com"], check=False)
    cmd([git, "config", "--global", "credential.helper", "manager"], check=False)


def github_user(gh):
    code, username, _ = out([gh, "api", "user", "-q", ".login"])
    if code != 0 or not username:
        raise SystemExit("[ERROR] 无法读取 GitHub 用户名，请先 gh auth login")
    return username


def get_origin(git, project):
    code, origin, _ = out([git, "-C", str(project), "remote", "get-url", "origin"])
    return origin if code == 0 and origin else None


def set_origin(git, project, origin):
    cmd([git, "-C", str(project), "remote", "add", "origin", origin], check=False)
    cmd([git, "-C", str(project), "remote", "set-url", "origin", origin], check=False)


def remove_origin(git, project):
    if get_origin(git, project):
        cmd([git, "-C", str(project), "remote", "remove", "origin"], check=False)


def parse_full_name(origin):
    if not origin:
        return None
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?/?$", origin.strip())
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def repo_exists(gh, full_name):
    if not full_name:
        return False
    code, _, _ = out([gh, "repo", "view", full_name])
    return code == 0


def origin_valid(git, gh, project):
    origin = get_origin(git, project)
    if not origin:
        return False
    full_name = parse_full_name(origin)
    if repo_exists(gh, full_name):
        print(f"[OK] origin 可用：{origin}")
        return True
    print(f"[WARN] origin 指向的仓库不存在：{origin}")
    return False


def create_or_link_repo(git, gh, project, args, force_replace=False):
    username = github_user(gh)
    repo = slug(project.name)
    full_name = f"{username}/{repo}"

    if force_replace:
        remove_origin(git, project)

    if not repo_exists(gh, full_name):
        is_public = choose_visibility(args)
        visibility = "--public" if is_public else "--private"
        print(f"[CREATE] {full_name} ({'public' if is_public else 'private'})")
        cmd([gh, "repo", "create", repo, visibility])
    else:
        print(f"[OK] GitHub 仓库已存在：{full_name}")

    origin = f"https://github.com/{full_name}.git"
    set_origin(git, project, origin)
    return origin


def commit_all(git, project, message):
    cmd([git, "-C", str(project), "status", "--short"], check=False)
    cmd([git, "-C", str(project), "add", "."])

    code, _, _ = out([git, "-C", str(project), "diff", "--cached", "--quiet"])
    if code == 0:
        print("[OK] 没有新改动，不重复 commit")
        return

    msg = message or f"Backup {datetime.now().strftime('%Y-%m-%d %H-%M')}"
    r = cmd([git, "-C", str(project), "commit", "-m", msg], check=False)

    if r.returncode != 0:
        print("[WARN] 需要设置 Git 名字和邮箱")
        name = input("Git user.name: ").strip()
        email = input("Git user.email: ").strip()
        if name:
            cmd([git, "config", "--global", "user.name", name])
        if email:
            cmd([git, "config", "--global", "user.email", email])
        cmd([git, "-C", str(project), "commit", "-m", msg])


def push(git, project):
    cmd([git, "-C", str(project), "push", "-u", "origin", "main"])


def open_repo(origin):
    if not origin:
        print("[WARN] 没有 origin，无法打开 GitHub 页面")
        return
    url = origin[:-4] if origin.endswith(".git") else origin
    url = url.replace("git@github.com:", "https://github.com/")
    print(f"[OPEN] {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--auto", action="store_true", help="跳过菜单，智能自动")
    parser.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    project = args.project.resolve()

    print("=" * 80)
    print(f"git_auto_push_menu.py | {VERSION}")
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
    add_ignore(project)
    ensure_github_login(gh, git)

    choice = "1" if args.auto else menu()

    if choice == "0":
        print("[EXIT]")
        return

    if choice == "4":
        open_repo(get_origin(git, project))
        return

    if choice == "2":
        if not origin_valid(git, gh, project):
            print("[STOP] 没有可用 origin。请选择 1 或 3 创建仓库")
            return
        commit_all(git, project, args.message)
        push(git, project)
        open_repo(get_origin(git, project))
        return

    if choice == "3":
        create_or_link_repo(git, gh, project, args, force_replace=True)
        commit_all(git, project, args.message)
        push(git, project)
        open_repo(get_origin(git, project))
        return

    if choice == "1":
        if not origin_valid(git, gh, project):
            create_or_link_repo(git, gh, project, args, force_replace=True)
        commit_all(git, project, args.message)
        push(git, project)
        open_repo(get_origin(git, project))
        return


if __name__ == "__main__":
    main()
