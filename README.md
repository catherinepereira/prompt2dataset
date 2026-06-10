# prompt2dataset

<img width="600" alt="prompt2dataset-cli" src="https://github.com/user-attachments/assets/4b715060-04c7-4335-b6b9-594743603cb2" />


Build labeled image and video datasets from a plain-English prompt.

```text
$ cd my-dataset
$ p2d add
What dataset do you want to build? > bird species native to the Pacific Northwest
```

prompt2dataset resolves your description into subjects via Claude, fetches media
from one or more sources, deduplicates, downloads, and writes a manifest.

## Installation

```bash
pip install prompt2dataset
```

`p2d add`, `review`, and `info` work with this base install. Training
requires PyTorch. Install the CPU or CUDA extras depending on your hardware:

```bash
pip install "prompt2dataset[train]"       # CPU
pip install "prompt2dataset[train-cuda]"  # CUDA (installs matching torch/torchvision)
```

## Setup

prompt2dataset needs an Anthropic API key. On first run it will prompt you and save the
key to a local `.env` file. Or set it yourself:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
P2D_CONTACT=you@example.com   # included in API request headers per Wikimedia's policy
```

## Usage

All commands operate on the current directory.

### `p2d add`

Prompts for a dataset description, asks whether you want images or video,
resolves subjects, and downloads media. Run it again in the same directory to
fetch additional subjects without re-downloading what's already there. The
media type is fixed when the dataset is created.

```bash
$ mkdir pacific-northwest-birds && cd pacific-northwest-birds
$ p2d add
```

### `p2d review`

Step through downloaded images and mark them valid or delete them.

```bash
$ p2d review
$ p2d review --misclassified   # only images that a trained model got wrong
```

Keys: **A** accept, **D** delete, **S** skip, **Q** quit.

### `p2d info`

Print dataset statistics and the subject list.

### `p2d train`

Image datasets only. Fine-tune a pretrained image classifier on the dataset. Uses
[torch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder) to find a good
learning rate automatically, then trains for N epochs and exports a TorchScript model.

```bash
$ p2d train
$ p2d train --model resnet50 --epochs 10
```

Options: `--epochs`, `--val-split`, `--img-size`, `--model` (mobilenet_v2, resnet18, resnet50).

## Data sources

| Source                | Best for                                                 |
|-----------------------|----------------------------------------------------------|
| **DuckDuckGo**        | Broad or niche subjects, recent events, pop culture      |
| **Wikimedia Commons** | Well-documented subjects with Wikipedia articles         |
| **iNaturalist**       | Animals, plants, fungi - research-grade, taxonomy-tagged |
| **Openverse**         | General subjects, scenes, cultural content               |
| **Wikimedia Commons (video)** | Freely licensed video clips (video datasets)     |

None require an API key. Sources are selected interactively when you run
`p2d add`, filtered to those that support the chosen media type.

## Output layout

```text
my-dataset/
  american-robin/
    american-robin_a3f1c8d2e9b4.jpg
    ...
  stellers-jay/
    ...
  .p2d/
    manifest.json       dataset metadata and item list
    labels.csv          filename, subject, source
    subjects.json       resolved subject list (cached)
    model.pt            TorchScript model (after p2d train)
    labels.json         class names in output order
    report.json         per-class precision/recall/F1
    misclassified.json  validation images the model got wrong
```

`manifest.json` is the authoritative record. Everything in `.p2d/` is
generated and can be reconstructed.

## Global flag

`--debug` enables verbose logging for all commands:

```bash
p2d --debug add
```

## License

MIT
