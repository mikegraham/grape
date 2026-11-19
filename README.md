# grape

Find images matching keywords using [CLIP](https://arxiv.org/abs/2103.00020)
(Radford et al., 2021). Like grep, but for images.

```
$ grape -R -s -n 5 --keywords sunset tests/fixtures
0.283  tests/fixtures/subdir/sunset.jpg
0.163  tests/fixtures/beach.jpg
0.145  tests/fixtures/omelette.jpg
0.144  tests/fixtures/lenna.png
0.128  tests/fixtures/toms_diner.jpg
```

The same search with `--view`:

<img src="https://raw.githubusercontent.com/mikegraham/grape/master/docs/screenshot_view.png" alt="grape --view screenshot" width="600">

## Install

```
pip install grape-image-search
```

Python 3.10+. Model weights (~3.5 GB for the default) download on first run.

### GPU

grape uses a GPU when PyTorch has one: NVIDIA, AMD or Intel. Even a
laptop's integrated GPU indexes ~20x faster than the CPU with the default
model. `-v` logs the device it picked; `GRAPE_DEVICE=cpu` forces the CPU.

- **NVIDIA**: the plain `pip install` above already gets PyTorch's CUDA build.
- **Intel (XPU)**: install the GPU driver and PyTorch's XPU build first
  ([PyTorch's Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)).
  `scripts/install_xpu.sh` sets up a venv this way.

  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
  pip install grape-image-search
  ```

- **AMD (ROCm, untested)**: install PyTorch's ROCm build first
  ([pytorch.org](https://pytorch.org/get-started/locally/)).

## Usage

Paths can be files, directories (with `-R`), or `-` to read from stdin.
At least one of `-k` or `--like` is required.

```bash
# describe what's in the picture (it matches what the photo shows, not
# words written in it)
grape -R --keywords sunset ~/Pictures

# find images similar to a reference
grape -R --like ref.jpg ~/Pictures

# combine text + reference, and penalize something
grape -R --keywords dog --exclude cat --like my_dog.jpg ~/Pictures

# top 5, with scores
grape -R -s -n 5 --keywords sunset ~/Pictures

# browse results in a GUI window
grape -R -n 200 --keywords sunset --view ~/Pictures

# pick images in the window (click to select); their paths print on close
grape -R -print0 --keywords blurry --view ~/Pictures | xargs -0 rm --

# read paths from stdin
find ~/Pictures -mtime -7 -type f | grape --keywords selfie -

# copy the top 10 cat photos to a folder
grape -R -print0 -n 10 --keywords cat ~/Pictures | xargs -0 cp -t ~/cats/
```

> [!NOTE]
> When using `--view`, keep `-n` at 1000 or less: the window gets slow
> with too many results.

Run `grape --help` for the full flag list.

## Worked examples

**Per-keyword breakdown** (`-v`) shows how each keyword and `--like`
reference contributes. The score on the header line is the mean.

```
$ grape -R -v -n 3 --keywords 'playful puppy,golden retriever' ~/Pictures
0.342  /home/me/Pictures/family/biscuit_park.jpg
  playful puppy: 0.301  golden retriever: 0.383
0.298  /home/me/Pictures/family/biscuit_couch.jpg
  playful puppy: 0.267  golden retriever: 0.329
0.274  /home/me/Pictures/family/old_biscuit.jpg
  playful puppy: 0.198  golden retriever: 0.350
```

`-v` also dumps every thread's stack on Ctrl-C, which shows where a run
is stuck.

**Aesthetic ranking** uses `--exclude` to subtract the mean score of a
contrast prompt. Scores are smaller here because both sides are competing.

```
$ grape -R -s -n 5 --ensemble-prompts '{}' \
    --keywords 'beautiful photo' --exclude 'ugly photo' ~/travel/
0.163  /home/me/travel/kyoto_arashiyama.jpg
0.139  /home/me/travel/iceland_sunset.jpg
0.127  /home/me/travel/nepal_trek.jpg
0.109  /home/me/travel/paris_evening.jpg
0.091  /home/me/travel/tokyo_skyline.jpg
```

## Scoring

Grape ranks images by cosine similarity between CLIP embeddings. Each
image's score is:

    mean(similarity to --keywords and --like) - mean(similarity to --exclude)

Cosine similarities are **not probabilities**. Meaningful matches typically
sit in the 0.15-0.35 range; even strong matches rarely exceed 0.35. Scores
are **not comparable across models** -- switch models and the numbers shift.

**`--like` matches semantics, not pixels.** Two different photos of dogs
playing fetch score high; a photo and its crop do too. For actual duplicate
detection, use a perceptual-hashing tool like
[czkawka](https://github.com/qarmin/czkawka).

## Caching

On by default at `~/.cache/grape/embeddings.db`
(`$XDG_CACHE_HOME/grape/embeddings.db`). Cached:

- Image embeddings (the expensive part)
- Text embeddings, keyed by prompt
- Model identity, so repeat runs skip the torch/open_clip import entirely
- Image-vs-non-image detection results, so videos and docs are skipped on rescan

A warm query scores thousands of images in under a second via a single
matrix multiply. Entries are keyed by absolute path, file stat (size, mtime,
inode), and model ID, so they auto-invalidate when files change.

Use `--no-cache` to disable or `--cache PATH` to relocate.

## Models

The default is
[SigLIP 2's ViT-L-16-SigLIP2-256](https://arxiv.org/abs/2502.14786)
(Tschannen et al., 2025) pretrained as `webli` -- the strongest text-to-image
retrieval available at this encoding cost. Pick any
[OpenCLIP](https://github.com/mlfoundations/open_clip) checkpoint with
`--model model_name/pretrained_tag`:

```bash
grape --model ViT-B-16-SigLIP2-256/webli -R --keywords sunset ~/Pictures    # 3.5x faster
grape --model ViT-SO400M-14-SigLIP2/webli -R --keywords sunset ~/Pictures   # strongest
```

`ViT-B-16-SigLIP2-256` indexes photos about 3.5x faster and is a 1.5 GB
download instead of 3.5 GB. Its matches are slightly worse: on the standard
ImageNet test it names about 3 in 100 more photos wrong than the default.

Embeddings are scoped per model; switching re-encodes everything.

Keywords are matched with
[prompt ensembling](https://arxiv.org/abs/2103.00020) (Radford et al., 2021,
Section 3.1.4 and Appendix A): each keyword expands into several templates
("itap of a {}.", "a bad photo of the {}.", ...) plus the bare keyword, each
is embedded, then the embeddings are averaged and renormalized. The default
set is open_clip's seven simple ImageNet templates plus "{}", which
[Volkov et al. (2025)](https://arxiv.org/abs/2509.09311) found best for
SigLIP 2. Override with `--ensemble-prompts`.

## Known issues

- `--view` can't display HEIC photos: the window's web engine doesn't
  decode them. Paths, scores and selection still work.
