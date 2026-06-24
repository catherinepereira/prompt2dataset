# prompt2dataset

<img width="600" alt="prompt2dataset-cli" src="https://github.com/user-attachments/assets/4b715060-04c7-4335-b6b9-594743603cb2" />


Build labeled image datasets from a plain-English prompt.

```text
$ cd my-dataset
$ p2d add
What dataset do you want to build? > bird species native to the Pacific Northwest
```

prompt2dataset resolves your description into subjects via a local Qwen model,
fetches images from one or more sources, deduplicates, downloads, and writes a manifest.

## Installation

```bash
pip install prompt2dataset
```

`p2d add`, `review`, `info`, and `dedup` work with this base install. Training
and `p2d outliers` require PyTorch:

```bash
pip install "prompt2dataset[train]"
```

This pulls a CPU build of torch. For a CUDA build, install from the matching
PyTorch index:

```bash
pip install "prompt2dataset[train]" --index-url https://download.pytorch.org/whl/cu126
```

Pick the index for your CUDA version (`cu121`, `cu124`, `cu126`, ...). See the
[PyTorch install guide](https://pytorch.org/get-started/locally/) for current options.

## Setup

prompt2dataset resolves subjects with a local [Ollama](https://ollama.com) model.
Install Ollama and pull the model:

```bash
ollama pull qwen2.5:3b-instruct
```

Optional environment variables (set in a local `.env` file):

```bash
# .env
OLLAMA_HOST=http://localhost:11434   # where Ollama is running
P2D_MODEL=qwen2.5:3b-instruct        # which model to resolve subjects with
P2D_CONTACT=you@example.com          # included in source API request headers per Wikimedia's policy
```

## Usage

All commands operate on the current directory.

### `p2d add`

Prompts for a dataset description, resolves subjects, and downloads images. Run
it again in the same directory to fetch additional subjects without
re-downloading what's already there.

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

### `p2d dedup`

Removes exact-duplicate images, found by hashing decoded pixels, so the same
image saved under a different filename or format is caught.

```bash
$ p2d dedup
$ p2d dedup --delete   # remove flagged files instead of marking invalid
```

### `p2d outliers`

Removes images that don't fit the rest of their subject. Each image is embedded
with a pretrained CNN, then DBSCAN flags those that don't cluster with the
others (scraping junk like charts or text-on-white). Needs the `[train]` extra.

```bash
$ p2d outliers
$ p2d outliers --delete    # remove flagged files instead of marking invalid
$ p2d outliers --eps 0.3   # looser clustering (flags fewer)
```

Both commands mark flagged images invalid in the manifest by default, so you can
inspect them with `p2d review` before they're gone. Pass `--delete` to remove the
files directly.

### `p2d info`

Print dataset statistics and the subject list.

### `p2d train`

Fine-tune a pretrained image classifier on the dataset. Uses
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
| **Bing**              | General web image search, high-volume results            |
| **Wikimedia Commons** | Well-documented subjects with Wikipedia articles         |
| **iNaturalist**       | Animals, plants, fungi - research-grade, taxonomy-tagged |
| **Openverse**         | General subjects, scenes, cultural content               |

None require an API key. Sources are selected interactively when you run
`p2d add`.

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
