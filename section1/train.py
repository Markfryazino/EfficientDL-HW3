import torch
import hydra
import wandb
import omegaconf

from hydra.utils import instantiate
from torch import nn
from tqdm.auto import tqdm
from typing import Union

from unet import Unet

from dataset import get_train_data
from scaler import CustomGradScaler


def zeros_num_in_grad(optimizer):
    num_zeros = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            num_zeros += (param.grad == 0).sum()
    return num_zeros


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer: torch.optim.Adam,
    scaler: Union[torch.cuda.amp.GradScaler, CustomGradScaler],
    fp16: bool,
    device: torch.device,
    log_steps: int,
    n_epoch: int
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    logging_loss = logging_zeros = logging_accuracy = 0
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        logging_zeros += zeros_num_in_grad(optimizer) / log_steps
        optimizer.zero_grad()

        accuracy = ((outputs > 0.5) == labels).float().mean()
        logging_loss += loss.item() / log_steps
        logging_accuracy += accuracy / log_steps

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")
        # увеличиваем wandb.step
        wandb.log({})

        if (i + 1) % log_steps == 0:
            wandb.log({
                "section1/loss": logging_loss,
                "section1/accuracy": logging_accuracy,
                "section1/zero number in gradients": logging_zeros,
                "section1/epoch": n_epoch
            }, step=wandb.run.step)

            logging_loss = logging_zeros = logging_accuracy = 0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name, config=config)

    device = torch.device(cfg.device)
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = instantiate(cfg.scaling)

    # проверка, что не будет обучения в fp32 со скейлингом
    assert cfg.fp16 or not scaler.is_enabled()

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, cfg.fp16, device=device, 
                    log_steps=cfg.wandb.log_steps, n_epoch=epoch)

    wandb.finish()


if __name__ == "__main__":
    train()