#!/usr/bin/env python3
import sys
import time
import subprocess
import os

# Change to repo directory
repo_path = r"c:\ml"
os.chdir(repo_path)

# Try to import GitPython with retries
max_retries = 15
retry_count = 0

print("Waiting for dependencies to install...")
while retry_count < max_retries:
    try:
        from git import Repo
        print("✓ GitPython imported successfully")
        break
    except ImportError:
        retry_count += 1
        if retry_count == max_retries:
            print("⚠ GitPython taking longer than expected, attempting alternative method...")
            break
        print(f"⏳ Waiting for GitPython... ({retry_count}/{max_retries})")
        time.sleep(3)

try:
    # Try using GitPython first
    try:
        from git import Repo as GitRepo
        repo = GitRepo(repo_path)
        print(f"✓ Repository found at: {repo.working_dir}")
        
        # Update remote URL
        new_url = "https://github.com/tamannafarooq/Movie-Recommendation.git"
        origin = repo.remote('origin')
        origin.set_url(new_url)
        print(f"✓ Remote URL updated to: {new_url}")
        
        # Add all files
        repo.git.add(A=True)
        print("✓ All files staged")
        
        # Commit
        try:
            repo.index.commit("Upload project files")
            print("✓ Changes committed successfully")
        except Exception as e:
            if "nothing to commit" in str(e):
                print("ℹ No new changes to commit")
            else:
                raise
        
        # Push to remote
        print("📤 Pushing to GitHub...")
        origin.push(refspec='main:main', force=True)
        print("✓ Successfully pushed to GitHub!")
        
    except (ImportError, NameError):
        # Fallback: Use git from PATH via subprocess
        print("Using git command-line tool...")
        
        # Update remote URL
        new_url = "https://github.com/tamannafarooq/Movie-Recommendation.git"
        subprocess.run(["git", "remote", "set-url", "origin", new_url], check=True)
        print(f"✓ Remote URL updated to: {new_url}")
        
        # Add all files
        subprocess.run(["git", "add", "-A"], check=True)
        print("✓ All files staged")
        
        # Commit
        try:
            subprocess.run(["git", "commit", "-m", "Upload project files"], check=True)
            print("✓ Changes committed successfully")
        except subprocess.CalledProcessError:
            print("ℹ No new changes to commit")
        
        # Push to remote
        print("📤 Pushing to GitHub...")
        subprocess.run(["git", "push", "-u", "origin", "main", "-f"], check=True)
        print("✓ Successfully pushed to GitHub!")
        print(f"✓ Repository URL: {new_url}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
