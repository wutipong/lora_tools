from pathlib import Path
from spandrel import ImageModelDescriptor, ModelLoader
from tqdm import tqdm
import click
import cv2
import logging
import numpy as np
import shutil
import torch

global main_dir
config_dir = ''


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0).cuda()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def project_dir_prepare(project_path):
    input_dir_path = project_path / "orig"
    output_dir_path = project_path / "dataset"

    if not input_dir_path.exists() and output_dir_path.exists():
        logging.warning(
            "'orig' path does not exists, but 'dataset' exists. Renaming 'dataset' to 'orig'."
        )
        shutil.move(output_dir_path, input_dir_path)

    elif not input_dir_path.exists:
        logging.error("'orig' directory not found. Abort.")
        return

    if not output_dir_path.exists():
        output_dir_path.mkdir()

    return input_dir_path, output_dir_path


def perform_resize_model(model_path, project_path):
    device = torch.device("cuda")
    model = ModelLoader().load_from_file(
        model_path
    )

    if not isinstance(model, ImageModelDescriptor):
        logging.error("invalid model")
        return

    model.cuda()

    input_dir_path, output_dir_path = project_dir_prepare(project_path)

    paths = [p for p in input_dir_path.iterdir()]
    try:
        for input_path in (pbar := tqdm(paths)):
            rel = input_path.relative_to(input_dir_path)
            output_path = output_dir_path / rel

            if input_path.is_dir():
                output_path.mkdir()
                continue

            pbar.set_description(str(rel))

            img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            img_tensor = image_to_tensor(img)
            img_tensor.to(device)

            out_tensor = model(img_tensor)
            out = tensor_to_image(out_tensor)

            cv2.imwrite(str(output_path), out)

            del img, img_tensor, out_tensor, out
    finally:
        logging.info("image upscale complete.")
        torch.cuda.empty_cache()
        del model


@click.group()
def cli():
    """Upscale dataset tool"""


@cli.command()
def list_model():
    """List available model files."""
    model_path = Path.cwd() / "models/upscale"
    paths = [p for p in model_path.iterdir()]

    for i in paths:
        rel = i.relative_to(model_path)
        if rel.suffix != '.pth' and rel.suffix != '.safetensor':
            continue
        print(rel)


@cli.command()
@click.option('--model', required=True, help='Model file name')
@click.argument('project')
def use_model(model, project):
    """Resize dataset images using upscale model."""
    model_path = Path.cwd() / "models/upscale" / model

    if not model_path.exists() or not model_path.is_file():
        logging.error(f"Unable to load model: {model_path}")
        return

    logging.info(f'Use Model: {model_path}')

    project_path = Path.cwd() / "workspace" / project

    if not project_path.exists() or project_path.is_file():
        logging.error(f"Unable to find project directory: {project_path}")
        return

    perform_resize_model(model_path, project_path)


if __name__ == "__main__":
    cli()
