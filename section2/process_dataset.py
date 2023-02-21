from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm

import hydra


def yield_tokens(dataset):
    for row in tqdm(dataset):
        yield row["tokens"]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def process(cfg):
    tokenizer = get_tokenizer("basic_english")
    data = \
        load_dataset("wikitext", "wikitext-103-raw-v1", split="train")\
        .filter(lambda x: len(x["text"]) > 0 and not x["text"].startswith(" ="))\
        .map(lambda x: {"tokens": tokenizer(x["text"])})

    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<pad>"])
    data = data.map(lambda x: {"indices": vocab(x["tokens"])})
    data.set_format("pt")
    data.save_to_disk(cfg.data_path)


if __name__ == "__main__":
    process()