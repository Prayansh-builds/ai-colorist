"""
push_to_github.py - Recursive sync to GitHub via REST API.
Uploads all code, workflows, and Android project files to GitHub without needing git binary.
Usage:
    python3 push_to_github.py <YOUR_GITHUB_TOKEN>
"""
import sys
import os
import json
import base64
import urllib.request
import urllib.error

REPO = "Prayansh-builds/ai-colorist"
BRANCH = "main"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IGNORE_EXTENSIONS = [".zip", ".pyc", ".swp"]
IGNORE_DIRS = [".git", "__pycache__", ".gradle", "build", ".idea"]

def get_current_sha(path, token):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "AI-Colorist-Sync"
    })
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data.get("sha")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

def upload_file(rel_path, token):
    abs_path = os.path.join(BASE_DIR, rel_path)
    with open(abs_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    sha = get_current_sha(rel_path, token)
    payload = {
        "message": f"Sync {rel_path} for AI Colorist full pipeline & APK build",
        "content": content_b64,
        "branch": BRANCH
    }
    if sha:
        payload["sha"] = sha

    url = f"https://api.github.com/repos/{REPO}/contents/{rel_path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Colorist-Sync"
        },
        method="PUT"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            action = "Updated" if sha else "Created"
            print(f"✓ {action} {rel_path}")
            return True
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"✗ Failed {rel_path}: HTTP {e.code} - {err[:150]}")
        return False

def collect_files():
    all_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if any(f.endswith(ext) for ext in IGNORE_EXTENSIONS) or f == "push_to_github.py":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, BASE_DIR)
            all_files.append(rel_path)
    return sorted(all_files)

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub token required.")
        print("Usage: python3 push_to_github.py <YOUR_GITHUB_TOKEN>")
        sys.exit(1)

    files = collect_files()
    print(f"Found {len(files)} files to sync to https://github.com/{REPO} on branch '{BRANCH}'...")
    success = 0
    for rel_path in files:
        if upload_file(rel_path, token):
            success += 1

    print(f"\nFinished! Successfully synced {success}/{len(files)} files.")
    print("GitHub Actions will now automatically trigger to build your Android APK!")

if __name__ == "__main__":
    main()
