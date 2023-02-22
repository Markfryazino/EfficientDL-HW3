import torch
import hydra
import wandb
import omegaconf
import numpy as np

from hydra.utils import instantiate
from tqdm.auto import tqdm
from time import perf_counter

from weird_gpt2 import WeirdGPT2, VOCAB_LENGTH
from dataset import UltraDuperBigBrainDataset, MAX_LENGTH


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_epoch(cfg) -> None:
    config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name, config=config, tags=["sec2-final"])

    device = torch.device(cfg.device)
    model = WeirdGPT2().to(device)

    dataset = instantiate(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = 1 if isinstance(dataset, UltraDuperBigBrainDataset) else cfg.batch_size,
        batch_sampler=dataset.get_batch_sampler(),
        num_workers=4,
        collate_fn=dataset.get_collator(),
        drop_last = False if isinstance(dataset, UltraDuperBigBrainDataset) else True,
        shuffle = None if isinstance(dataset, UltraDuperBigBrainDataset) else True,
    )

    # для прогрева GPU один раз прогоняем через модель батч максимального размера
    x = torch.randint(1, VOCAB_LENGTH - 1, (cfg.batch_size, MAX_LENGTH))
    model(x.to(device))
    torch.cuda.synchronize(device)

    times = []

    for x, _ in tqdm(dataloader, total=len(dataloader)):
        start_time = perf_counter()
        model(x.to(device))
        torch.cuda.synchronize(device)
        times.append(perf_counter() - start_time)

    wandb.run.summary.update({
        "section2/min_time": np.min(times),
        "section2/max_time": np.max(times),
        "section2/mean_time": np.mean(times),
        "section2/median_time": np.median(times)
    })

    wandb.finish()

if __name__ == "__main__":
    run_epoch()
