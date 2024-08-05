import click
from pathlib import Path
import logging
import cv2
from simple_slurm import Slurm


@click.command()
@click.argument('project')
def cli(project):
    """Queue training job."""

    project_path = Path.cwd() / "workspace" / project

    if not project_path.exists() or project_path.is_file():
        logging.error(f"Unable to find project directory: {project_path}")
        return

    dataset_config_path = project_path / "dataset_config.toml"
    training_config_path = project_path / "training_config.toml"

    if not dataset_config_path.exists():
        logging.error(f"Unable to dataset_config.toml: {project_path}")
        return

    if not training_config_path.exists():
        logging.error(f"Unable to training_config.toml: {project_path}")
        return

    slurm = Slurm(
        gres=['gpu:1'],
        job_name=f'train-sdxl-{project}',
        output=f'{
            Path.cwd()}/logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    slurm.add_cmd('source', f'{Path.cwd()}/venv/bin/activate')

    slurm.add_cmd('accelerate', 'launch', '--quiet',
                  f'--config_file={Path.cwd()}/accelerate_config.yaml',
                  f'--num_cpu_threads_per_process=1',
                  f'{Path.cwd()}/sd-scripts/sdxl_train_network.py',
                  f'--dataset_config={project_path}/dataset_config.toml',
                  f'--config_file={project_path}/training_config.toml'
                  )

    slurm.sbatch()


if __name__ == "__main__":
    cli()
