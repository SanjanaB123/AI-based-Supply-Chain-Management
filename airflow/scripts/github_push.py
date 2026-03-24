import os
import sys
from github import Github, GithubException

def push_to_github(file_path, repo_name, token, commit_message):
    gh = Github(token)
    try:
        repo = gh.get_repo(repo_name)
        # Convert path to be relative to base dir (e.g. /opt/airflow)
        base_dir = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, base_dir)
        
        with open(file_path, "rb") as f:
            content = f.read()

        try:
            contents = repo.get_contents(rel_path)
            repo.update_file(contents.path, commit_message, content, contents.sha)
            print(f"Updated {rel_path} on GitHub.")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(rel_path, commit_message, content)
                print(f"Created {rel_path} on GitHub.")
            else:
                raise e
        return True
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        return False

def delete_from_github(file_path, repo_name, token, commit_message):
    gh = Github(token)
    try:
        repo = gh.get_repo(repo_name)
        base_dir = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, base_dir)
        contents = repo.get_contents(rel_path)
        repo.delete_file(contents.path, commit_message, contents.sha)
        print(f"Deleted {rel_path} from GitHub.")
        return True
    except GithubException as e:
        if e.status == 404:
            print(f"File {file_path} not found on GitHub, skipping deletion.")
            return True
        else:
            print(f"Error deleting from GitHub: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def list_github_files(repo_name, token, path=""):
    gh = Github(token)
    try:
        repo = gh.get_repo(repo_name)
        contents = repo.get_contents(path)
        print(f"Contents of {repo_name}/{path}:")
        for content in contents:
            print(f"- {content.path} ({content.type})")
        return True
    except Exception as e:
        print(f"Error listing files: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python github_push.py <command> [args...]")
        print("Commands: push, delete, list")
        sys.exit(1)

    command = sys.argv[1]
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPO", "SanjanaB123/AI-based-Supply-Chain-Management")

    if not token:
        print("Error: GITHUB_TOKEN not found in environment.")
        sys.exit(1)

    if command == "list":
        path = sys.argv[2] if len(sys.argv) > 2 else ""
        list_github_files(repo_name, token, path)
        sys.exit(0)

    if len(sys.argv) < 4:
        print(f"Usage: python github_push.py {command} <commit_message> <file_path1> ...")
        sys.exit(1)

    commit_message = sys.argv[2]
    file_paths = sys.argv[3:]

    success = True
    for file_path in file_paths:
        if command == "push":
            if not push_to_github(file_path, repo_name, token, commit_message):
                success = False
        elif command == "delete":
            if not delete_from_github(file_path, repo_name, token, commit_message):
                success = False
    
    if not success:
        sys.exit(1)
