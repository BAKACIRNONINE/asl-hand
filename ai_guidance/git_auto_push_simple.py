r"""
git_auto_push_simple.py

简化版：
1. git init
2. 自动补 .gitignore
3. commit
4. 如果没有可用 origin，就按文件夹名创建 GitHub 仓库
5. push
6. 打开仓库网页

运行：
python "ai_guidance/git_auto_push_simple.py"
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


def find_exe(name, candidates):
    found = shutil.which(name)
    if found:
        return found
    for p in candidates:
        if Path(p).exists():
            return p
    raise SystemExit(f"{name} not found")


def slug(name):
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-._")
    return name or "project-backup"


def output(args, cwd=None):
    r = cmd(args, cwd=cwd, check=False, capture=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def add_ignore(project):
    p = project / ".gitignore"
    text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if "AI_BACKUP_IGNORE_BEGIN" not in text:
        p.write_text((text.rstrip() + "\n\n" + IGNORE_BLOCK + "\n").lstrip(), encoding="utf-8")


def get_origin(git, project):
    code, out, _ = output([git, "-C", str(project), "remote", "get-url", "origin"])
    return out if code == 0 and out else None


def remove_origin(git, project):
    if get_origin(git, project):
        cmd([git, "-C", str(project), "remote", "remove", "origin"], check=False)


def user_login(gh):
    code, out, _ = output([gh, "api", "user", "-q", ".login"])
    if code != 0 or not out:
        cmd([gh, "auth", "login"])
        code, out, _ = output([gh, "api", "user", "-q", ".login"])
    return out


def repo_exists(gh, full_name):
    code, _, _ = output([gh, "repo", "view", full_name])
    return code == 0


def origin_repo_exists(gh, origin):
    if not origin:
        return False
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?", origin)
    if not m:
        return False
    return repo_exists(gh, f"{m.group(1)}/{m.group(2)}")


def choose_visibility(args):
    if args.public:
        return True
    if args.private:
        return False
    print("\n选择仓库权限：")
    print("1. private 私有")
    print("2. public 公开")
    return (input("输入 1/2，回车默认 1: ").strip() or "1") == "2"


def commit_all(git, project, message):
    cmd([git, "-C", str(project), "add", "."])
    code, _, _ = output([git, "-C", str(project), "diff", "--cached", "--quiet"])
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
    parser.add_argument("-m", "--message", default=None)
    args = parser.parse_args()

    project = args.project.resolve()
    git = find_exe("git", [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ])
    gh = find_exe("gh", [
        r"C:\Program Files\GitHub CLI\gh.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "GitHub CLI" / "gh.exe"),
    ])

    print(f"[PROJECT] {project}")

    add_ignore(project)

    cmd([git, "-C", str(project), "init"])
    cmd([git, "-C", str(project), "checkout", "-B", "main"], check=False)

    commit_all(git, project, args.message)

    cmd([gh, "auth", "status"], check=False)
    login = user_login(gh)

    origin = get_origin(git, project)
    if origin and not origin_repo_exists(gh, origin):
        print(f"[WARN] origin 不存在，删除旧 origin：{origin}")
        remove_origin(git, project)
        origin = None

    if not origin:
        repo = slug(project.name)
        full_name = f"{login}/{repo}"
        visibility_public = choose_visibility(args)
        visibility = "--public" if visibility_public else "--private"

        if not repo_exists(gh, full_name):
            print(f"[CREATE] {full_name}")
            cmd([gh, "repo", "create", repo, visibility])
        else:
            print(f"[OK] GitHub 仓库已存在：{full_name}")

        origin = f"https://github.com/{full_name}.git"
        cmd([git, "-C", str(project), "remote", "add", "origin", origin])

    cmd([git, "-C", str(project), "push", "-u", "origin", "main"])
    open_repo(origin)
    print("[DONE]")


if __name__ == "__main__":
    main()
