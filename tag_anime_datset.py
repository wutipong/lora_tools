import click
from pathlib import Path
import logging
import subprocess


@click.command()
@click.option('--model', default='SmilingWolf/wd-swinv2-tagger-v3')
@click.argument('project')
def cli(model, project):
    """Tags dataset using Waifu Diffusion model."""

    project_path = Path.cwd() / "workspace" / project
    dataset_path = project_path / "dataset"

    if not project_path.exists() or project_path.is_file():
        logging.error(f"Unable to find project directory: {project_path}")
        return

    dataset_path = project_path / "dataset"

    if not dataset_path.exists() or dataset_path.is_file():
        logging.error(f"Unable to find dataset directory: {dataset_path}")
        return

    script_dir = Path.cwd() / "sd-scripts"

    subprocess.call(["python",
                     f"{script_dir / "finetune/tag_images_by_wd14_tagger.py"}",
                     "--onnx",
                     f"--repo_id={model}",
                     f"--model_dir={Path.cwd()/"models/tagger"}",
                     "--thresh=0.35",
                     "--batch_size=8",
                     "--caption_extension=.txt",
                     f"{dataset_path}"])


if __name__ == "__main__":
    cli()
