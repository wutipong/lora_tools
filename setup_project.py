import click
from pathlib import Path
import logging
import mimetypes
import cv2


@click.command()
@click.argument('project')
def cli(project):
    """Create project configuration files."""

    mimetypes.init()

    project_path = Path.cwd() / "workspace" / project
    dataset_path = project_path / "dataset"

    if not project_path.exists() or project_path.is_file():
        logging.error(f"Unable to find project directory: {project_path}")
        return

    dataset_path = project_path / "dataset"

    if not dataset_path.exists() or dataset_path.is_file():
        logging.error(f"Unable to find dataset directory: {dataset_path}")
        return

    paths = [p for p in dataset_path.iterdir() if p.is_file()]
    count = 0

    for path in paths:
        if cv2.haveImageReader(str(path)):
            count = count + 1

    print(f"found {count} images.")

    repeats = click.prompt(
        'Enter the number of repeats. Images multiplied by their repeats should be between 200 and 400', type=int)

    dataset_config_path = project_path / "dataset_config.toml"
    dataset_config = open(dataset_config_path, "w")
    dataset_config.write(
        f"""[[datasets]]

[[datasets.subsets]]
num_repeats = {repeats}
image_dir = "{dataset_path}"

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
lr_warmup_steps = 56
optimizer_type = "Prodigy"
optimizer_args = [ "weight_decay=0.1", "betas=[0.9,0.99]",]

[training_arguments]
pretrained_model_name_or_path = "hollowstrawberry/67AB2F"
vae = "stabilityai/sdxl-vae"
max_train_epochs = 10
train_batch_size = 2
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
output_dir = "{project_path/ "output"}"
log_prefix = "{project}"
logging_dir = "{project_path/ "logs"}"
    """)

    dataset_config.close()


if __name__ == "__main__":
    cli()
