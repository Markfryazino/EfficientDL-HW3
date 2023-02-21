import torch
import hydra

from hydra.utils import instantiate
from tqdm.auto import tqdm

from weird_gpt2 import WeirdGPT2
from dataset import UltraDuperBigBrainDataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_epoch(cfg) -> None:
    model = WeirdGPT2().to(cfg.device)

    dataset = instantiate(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = None if isinstance(dataset, UltraDuperBigBrainDataset) else cfg.batch_size,
        batch_sampler=dataset.get_batch_sampler(),
        num_workers=4,
        collate_fn=dataset.get_collator(),
        drop_last = None if isinstance(dataset, UltraDuperBigBrainDataset) else True,
    )

    for x, target in tqdm(dataloader):
        model(x.to(cfg.device))
    

if __name__ == "__main__":
    run_epoch()
