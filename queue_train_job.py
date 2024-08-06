from environs import Env
from pathlib import Path
from simple_slurm import Slurm
import click
import stringcase


@click.command()
@click.option('--dump', is_flag=True, help='Dump Slurm command.')
def cli(dump):
    """Queue training job."""

    queue_path = Path.cwd() / "queue"

    for project_path in queue_path.iterdir():
        project = project_path.name

        if project.startswith('.'):
            continue

        click.secho(f'Processing: {project}', fg='green')

        dataset_config_path = project_path / "dataset_config.toml"
        training_config_path = project_path / "training_config.toml"

        if not dataset_config_path.exists():
            click.secho(f"Unable to dataset_config.toml: {
                        project_path}", fg='red')
            continue

        if not training_config_path.exists():
            click.secho(f"Unable to training_config.toml: {
                project_path}", fg='red')
            continue

        slurm = Slurm(
            gres=['gpu:1'],
            job_name=f'train-sdxl-{project}',
            output=f'{
                Path.cwd()}/logs/{Slurm.JOB_ARRAY_MASTER_ID}_{stringcase.spinalcase(project)}_{Slurm.JOB_ARRAY_ID}.out',
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
                    'cp', '-uf', f'{project_path}/output/{project}.safetensors', path)
        except:
            click.secho('No ouput path specified. Skipped.', fg='yellow')

        slurm.add_cmd('mv', f'{project_path}', f"{Path.cwd()}/finished")

        if dump:
            click.echo(slurm)
        else:
            slurm.sbatch()


if __name__ == "__main__":
    cli()
