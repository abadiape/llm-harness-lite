"""Agent loop with local python tool which scanns for large files + optionally removing them."""
import json
import math
import os
from pathlib import Path
from typing import TypedDict

import yaml
from dotenv import load_dotenv

import llm_harness_lite

load_dotenv()

cfg = yaml.safe_load(open("config.yaml"))  # noqa: SIM115


class FileEntry(TypedDict):
    path: str
    size_bytes: int
    size_mib: float


class ScanReport(TypedDict):
    scanned_directory: str
    total_matching_files: int
    top_files: list[FileEntry]


reg = llm_harness_lite.ToolRegistry()
scan_report: ScanReport | None = None

# --- 1. Tool Implementations ---
@llm_harness_lite.tool(reg)
def find_largest_files(directory: str, min_size_mib: float) -> str:
    """
    Find files at least min_size_mib in size and return the 10 largest.
    """
    global scan_report
    scan_report = None

    try:
        target_path = resolve_directory(directory)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    if isinstance(min_size_mib, bool) or not isinstance(min_size_mib, (int, float)):
        return json.dumps(
            {"error": "min_size_mib must be a finite number greater than or equal to 0."}
        )
    try:
        min_size_mib = float(min_size_mib)
    except OverflowError:
        return json.dumps(
            {"error": "min_size_mib must be a finite number greater than or equal to 0."}
        )
    if not math.isfinite(min_size_mib) or min_size_mib < 0:
        return json.dumps(
            {"error": "min_size_mib must be a finite number greater than or equal to 0."}
        )

    files_info = []
    min_bytes = min_size_mib * (1024 * 1024)

    for root, _, filenames in os.walk(target_path):
        for filename in filenames:
            filepath = Path(root) / filename
            try:
                if filepath.is_symlink():
                    continue
                size_bytes = filepath.stat().st_size
                if size_bytes >= min_bytes:
                    files_info.append({
                        "path": str(filepath),
                        "size_bytes": size_bytes,
                        "size_mib": round(size_bytes / (1024 * 1024), 2)
                    })
            except (PermissionError, OSError):
                continue

    files_info.sort(key=lambda x: x["size_bytes"], reverse=True)
    top_files = files_info[:10]

    scan_report = {
        "scanned_directory": str(target_path),
        "total_matching_files": len(files_info),
        "top_files": top_files,
    }
    return json.dumps(scan_report, indent=2)


# Only the scan function is registered as an agent tool.
def delete_file(filepath: str) -> str:
    """Permanently delete a specific file."""
    try:
        target_path = Path(filepath).expanduser()
        if target_path.is_symlink():
            return json.dumps({
                "status": "error",
                "message": f"Refusing to delete a symbolic link: {target_path}",
            })
        target_path = target_path.resolve(strict=True)
        if not target_path.is_file():
            return json.dumps({"status": "error", "message": f"Not a file: {target_path}"})
        file_size = target_path.stat().st_size
        target_path.unlink()
        return json.dumps({
            "status": "success",
            "deleted_file": str(target_path),
            "freed_bytes": file_size,
        })
    except (OSError, RuntimeError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})

def resolve_directory(raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Enter a non-empty directory path.")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve that path: {exc}") from exc
    if not path.is_dir():
        raise ValueError(f"That path is not a directory: {path}")
    return path


def prompt_for_directory() -> Path:
    while True:
        raw_path = input("Directory to scan (absolute, relative, or ~ path): ").strip()
        try:
            return resolve_directory(raw_path)
        except ValueError as exc:
            print(f"Invalid directory: {exc}")


def prompt_for_min_size_mib() -> float:
    while True:
        raw_size = input("Minimum file size in MiB (0 or greater): ").strip()
        try:
            min_size_mib = float(raw_size)
        except ValueError:
            print("Invalid size: enter a number such as 1 or 1.5.")
            continue
        if not math.isfinite(min_size_mib) or min_size_mib < 0:
            print("Invalid size: enter a finite number greater than or equal to 0.")
            continue
        return min_size_mib


def confirm(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [yes/no]: ").strip().casefold()
        if answer in {"yes", "y"}:
            return True
        if answer in {"no", "n"}:
            return False
        print("Please answer yes or no.")


def print_file_list(files: list[FileEntry]) -> None:
    for number, file_info in enumerate(files, start=1):
        print(
            f"{number}. {file_info['size_mib']:.2f} MiB — "
            f"{file_info['path']}"
        )


def delete_and_report(filepath: str) -> None:
    outcome = json.loads(delete_file(filepath))
    if outcome["status"] == "success":
        print(
            f"Deleted {outcome['deleted_file']} "
            f"({outcome['freed_bytes']} bytes)."
        )
    else:
        print(f"Could not delete {filepath}: {outcome['message']}")


def run_deletion_menu(files: list[FileEntry]) -> None:
    while True:
        print("\nDeletion options for the listed files:")
        print("1. Delete all listed files")
        print("2. Choose a file by number")
        print("3. Quit without (any or) more deletions")
        choice = input("Choose 1, 2, or 3: ").strip().casefold()

        if choice in {"1", "all"}:
            print("\nAll files selected for deletion:")
            print_file_list(files)
            if confirm("Delete all listed files?"):
                for file_info in files:
                    delete_and_report(file_info["path"])
                return
            print("No files were deleted.")
        elif choice in {"2", "choose"}:
            print("\nChoose from the listed files:")
            print_file_list(files)
            while True:
                selection = input(
                    "Enter a file number, or q to quit: "
                ).strip().casefold()
                if selection in {"q", "quit", "exit"}:
                    return
                try:
                    number = int(selection)
                except ValueError:
                    print("Enter a listed number or q to quit.")
                    continue
                if not 1 <= number <= len(files):
                    print(f"Choose a number from 1 to {len(files)}, or q to quit.")
                    continue

                file_info = files[number - 1]
                filepath = file_info["path"]
                if confirm(f"Delete this file?\nFull path: {filepath}"):
                    delete_and_report(filepath)
                else:
                    print("No file deleted. Choose another number or q to quit.")
        elif choice in {"3", "q", "quit", "exit"}:
            return
        else:
            print("Choose 1, 2, or 3.")


directory = prompt_for_directory()
min_size_mib = prompt_for_min_size_mib()
llm = llm_harness_lite.create_llm(
    cfg["llm"]["provider"],
    cfg["llm"]["model"],
    base_url=cfg["llm"].get("base_url"),
    timeout=cfg["llm"].get("timeout", 300.0),
)
agent = llm_harness_lite.Agent(
    llm,
    reg,
    max_steps=cfg["agent"]["max_steps"],
    budget_usd=cfg["agent"]["budget_usd"],
)
goal = (
    "Call find_largest_files with exactly these values: "
    f"directory={json.dumps(str(directory))}, min_size_mib={min_size_mib:g}. "
    "Report the scan results."
)
res = agent.run(goal)
print(res.answer)
print(
    f"[steps {res.steps} | ${res.usage.cost_usd:.5f} | "
    f"{res.usage.input_tokens} input tokens | {res.usage.output_tokens} output tokens | "
    f"{res.usage.latency_s:.2f}s latency]"
)
if scan_report is None:
    print("The scan tool did not return a usable result; no files were changed.")
elif not scan_report["top_files"]:
    print("No matching files were found; there is nothing to delete.")
else:
    print("\nTop files eligible for deletion:")
    print_file_list(scan_report["top_files"])
    run_deletion_menu(scan_report["top_files"])