import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download a folder from a Hugging Face repository."
    )

    parser.add_argument(
        "--username", default="MrKustic", type=str, help="HF user or organization"
    )
    parser.add_argument("--project", default="asr", type=str, help="HF repository name")
    parser.add_argument(
        "--folder", required=True, type=str, help="Folder path inside the HF repository"
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=str,
        help="Destination local directory. The repository structure is preserved",
    )
    args = parser.parse_args()

    snapshot_download(
        repo_id=f"{args.username}/{args.project}",
        allow_patterns=[f"{args.folder}/*"],
        local_dir=args.local_dir,
    )


if __name__ == "__main__":
    main()
