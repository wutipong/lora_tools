from environs import Env
from pathlib import Path
from simple_slurm import Slurm
import click
import logging


@click.command()
@click.option('--dump', is_flag=True, help='Dump Slurm command.')
@click.argument('project')
def cli(dump, project):
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

    env = Env()

    try:
        lora_output_paths = env.list('LORA_TOOL_OUTPUT_PATHS')

        for path in lora_output_paths:
            slurm.add_cmd(
                'cp', f'{project_path}/output/{project}.safetensors', path)
    except:
        click.echo('No ouput path specified. Skipped.')

    if dump:   
        click.echo(slurm)
    else:
        slurm.sbatch()


if __name__ == "__main__":
    cli()
