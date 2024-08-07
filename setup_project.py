from pathlib import Path
import click
import cv2
import mimetypes
import shutil

TRAINING_BATCH_SIZE = 2
MAX_TRAINING_EPOCH = 10
LR_WARMUP_RATIO = 0.05


@click.command()
@click.argument('project')
def cli(project):
    """Create project configuration files."""

    mimetypes.init()

    project_path = Path.cwd() / "workspace" / project
    queue_project_path = Path.cwd() / "queue" / project

    if not project_path.exists() or project_path.is_file():
        click.secho(f"Unable to find project directory: {project_path}", fg='red')
        return

    dataset_path = project_path / "dataset"

    if not dataset_path.exists() or dataset_path.is_file():
        click.secho(f"Unable to find dataset directory: {dataset_path}", fg='red')
        return

    paths = [p for p in dataset_path.iterdir() if p.is_file()]
    count = 0

    for path in paths:
        if cv2.haveImageReader(str(path)):
            count = count + 1

    click.echo(f"found {count} images.")

    repeats = click.prompt(
        'Enter the number of repeats. Images multiplied by their repeats should be between 200 and 400', type=int)

    pre_steps_per_epoch = count * repeats
    steps_per_epoch = pre_steps_per_epoch/TRAINING_BATCH_SIZE
    total_steps = int(MAX_TRAINING_EPOCH*steps_per_epoch)
    lr_warmup_steps = int(total_steps*LR_WARMUP_RATIO)

    queue_dataset_path = queue_project_path / "dataset"
    dataset_config_path = project_path / "dataset_config.toml"
    dataset_config = open(dataset_config_path, "w")
    dataset_config.write(
        f"""[[datasets]]

[[datasets.subsets]]
num_repeats = {repeats}
image_dir = "{queue_dataset_path}"

[general]
resolution = 1024
shuffle_caption = true
keep_tokens = 1
flip_aug = false
caption_extension = ".txt"
enable_bucket = true
bucket_no_upscale = false
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096
    """)

    dataset_config.close()

    training_config_path = project_path / "training_config.toml"
    training_config = open(training_config_path, "w")
    training_config.write(
        f"""[network_arguments]
unet_lr = 0.75
text_encoder_lr = 0.75
network_dim = 16
network_alpha = 16
network_module = "networks.lora"
network_args = [ "conv_dim=16", "conv_alpha=8",]
network_train_unet_only = false

[optimizer_arguments]
learning_rate = 0.75
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3
lr_warmup_steps = {lr_warmup_steps}
optimizer_type = "Prodigy"
optimizer_args = [ "weight_decay=0.1", "betas=[0.9,0.99]",]

[training_arguments]
pretrained_model_name_or_path = "hollowstrawberry/67AB2F"
vae = "stabilityai/sdxl-vae"
max_train_epochs = {MAX_TRAINING_EPOCH}
train_batch_size = {TRAINING_BATCH_SIZE}
seed = 42
max_token_length = 225
xformers = false
sdpa = true
min_snr_gamma = 8.0
lowram = false
no_half_vae = true
gradient_checkpointing = true
gradient_accumulation_steps = 1
max_data_loader_n_workers = 8
persistent_data_loader_workers = true
mixed_precision = "bf16"
full_bf16 = true
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = false
min_timestep = 0
max_timestep = 1000
prior_loss_weight = 1.0
multires_noise_iterations = 6
multires_noise_discount = 0.3

[saving_arguments]
save_precision = "fp16"
save_model_as = "safetensors"
save_every_n_epochs = 1
save_last_n_epochs = 10
output_name = "{project}"
output_dir = "{queue_project_path}/output"
log_prefix = "{project}"
logging_dir = "{queue_project_path}/logs"
""")

    dataset_config.close()

    shutil.move(project_path, queue_project_path)

if __name__ == "__main__":
    cli()
