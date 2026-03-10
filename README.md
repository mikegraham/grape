# grape

Find images matching keywords using [CLIP](https://arxiv.org/abs/2103.00020).
Like grep, but for images.

```
$ grape -R -s -k sunset ~/Pictures
0.327  ~/Pictures/vacation/beach_golden_hour.jpg
0.285  ~/Pictures/vacation/pier_evening.jpg
0.241  ~/Pictures/hiking/mountain_view.jpg
```

Grape caches everything aggressively: image embeddings, text embeddings,
model information, and image detection results are all stored in a local
SQLite database. After the first run, repeat queries are very fast.

## Install

```
pip install .
```

Requires Python 3.10+.
Model weights (~1.7 GB for the default EVA02-L-14 model) are downloaded
on first run.

`--view` support (native results window) is installed by default.

If you want a minimal/headless install without GUI deps:
```
pip install . --no-deps
pip install open-clip-torch torch dask Pillow platformdirs tqdm transformers sentencepiece
```

## Quick start

```bash
# search by keyword
grape --keywords sunset photo.jpg

# multiple keywords, ranked by average similarity
grape --keywords 'cat,dog' *.jpg

# recursive directory search
grape -R --keywords 'golden retriever' ~/Pictures

# find images similar to a reference image
grape -R --like reference.jpg ~/Pictures

# combine text and image queries
grape -R --keywords dog --like my_dog.jpg ~/Pictures
```

## Examples

```bash
# show scores (--scores) or full per-keyword breakdown (--verbose)
grape --scores --keywords sunset *.jpg
grape --verbose --keywords 'cat,dog,bird' *.jpg

# top 5 results above a similarity threshold
grape -R --top 5 --threshold 0.25 --keywords sunset ~/Pictures

# prefer "dog", penalize "cat" (score = include_mean - exclude_mean)
grape -R --keywords dog --exclude cat ~/Pictures

# multiple reference images
grape -R --like ref1.jpg --like ref2.jpg ~/Pictures

# browse results in a GUI window
grape -R --view --keywords sunset ~/Pictures

# use a different model
grape -R --model ViT-L-14/laion2b_s32b_b82k --keywords sunset ~/Pictures

# copy the top 10 cat photos to a folder
grape -R --quiet -print0 --top 10 --keywords cat ~/Pictures \
  | xargs -0 cp -t ~/cats/

# open the best match directly
grape -R --top 1 --keywords 'golden gate bridge' ~/Pictures \
  | xargs open

# interactive selection with fzf
grape -R --keywords dog ~/Pictures \
  | fzf --preview 'chafa {}'

# find semantically similar images (same subject/scene, not pixel-level)
grape -R --scores --like photo.jpg ~/Pictures

# find similar images, excluding a specific style
grape -R --scores --like vacation.jpg --exclude 'indoor,night' ~/Pictures

# only high-res images (pre-filter with exiftool, then score)
exiftool -if '$ImageWidth >= 1920 and $ImageHeight >= 1080' \
  -printFormat '$Directory/$FileName' ~/Pictures/*.jpg \
  | xargs grape --keywords sunset --scores

# only files modified in the last 7 days
find ~/Pictures -mtime -7 -type f | xargs grape --keywords selfie --scores
```

**Note on `--like` and duplicates:** `--like` finds *semantically* similar
images (same subject, scene, or style) -- not pixel-level duplicates. A
photo and its cropped version will score high, but so will two completely
different photos of dogs playing fetch. For finding actual duplicates, near-duplicates,
and resized copies, use a perceptual hashing tool like
[czkawka](https://github.com/qarmin/czkawka) instead.

## Options

### Query

| Flag | Description |
|------|-------------|
| `-k KEYWORDS` | Comma-separated keywords to match |
| `-x KEYWORDS` | Anti-match keywords to penalize |
| `--like IMAGE` | Reference image for similarity search (repeatable) |

At least one of `-k` or `--like` is required.

### Input

| Flag | Description |
|------|-------------|
| `-R` | Search directories recursively, following symlinks |

### Filtering

| Flag | Description |
|------|-------------|
| `-t SCORE` | Only show results with score >= SCORE |
| `-n N` | Show only top N results |

### Output

| Flag | Description |
|------|-------------|
| `-s` | Show scores |
| `-v` | Per-keyword score breakdown (implies `-s`); also enables debug logging |
| `-q` | Suppress status messages on stderr |
| `-print0` | NUL-separated output for `xargs -0` |
| `--view` | Browse results in a GUI window |

### Configuration

| Flag | Description |
|------|-------------|
| `--cache PATH` | SQLite cache file (default: `~/.cache/grape/embeddings.db`) |
| `--no-cache` | Disable caching entirely |
| `--model MODEL` | OpenCLIP model/pretrained tag (default: `EVA02-L-14/merged2b_s4b_b131k`) |

Image detection is content-based (PIL), not extension-based.
Output paths are shell-quoted by default.

## Caching

Caching is on by default. The cache lives at `~/.cache/grape/embeddings.db`
(or `$XDG_CACHE_HOME/grape/embeddings.db`) and stores:

- **Image embeddings**: the expensive part. Encoding an image takes
  ~80ms on CPU; with a warm cache, scoring is a single matrix multiply.
- **Text embeddings**: prompt-level, so "a photo of a dog" is cached
  once and reused across queries that include "dog".
- **Model identity**: cached so repeat runs skip the torch/open_clip
  import entirely (~1.5s saved).
- **Image detection**: files identified as non-images (videos, documents,
  etc.) are remembered and skipped on future scans.

Cache entries are keyed by absolute path, file stat (size, mtime, inode),
and model identifier. They auto-invalidate when a file changes. A fully
warm query over thousands of images typically completes in under 50ms.

Use `--no-cache` to disable, or `--cache PATH` to use a different file.

## Scoring

Images are ranked by cosine similarity between CLIP embeddings.
Scores typically land in the 0.15--0.35 range for meaningful matches --
they are similarities, not probabilities, and even strong matches
rarely exceed 0.35.

With `-x`, the score becomes `mean(include) - mean(exclude)`.
With `--like`, reference image similarities are included in the mean
alongside text keywords.

Scores are **not comparable across models**.

## Models

The default model is
[EVA02-L-14](https://arxiv.org/abs/2303.15389) (`merged2b_s4b_b131k`),
which offers strong zero-shot accuracy at moderate compute cost.

Use `--model` to pick any model from the
[OpenCLIP](https://github.com/mlfoundations/open_clip) pretrained
registry:

```bash
# lighter, faster
grape --model ViT-B-32/laion2b_s34b_b79k -k sunset -R ~/Pictures

# classic strong baseline
grape --model ViT-L-14/laion2b_s32b_b82k -k sunset -R ~/Pictures
```

The format is `model_name/pretrained_tag`. Larger models produce better
embeddings but are slower to encode. Cached embeddings are scoped per
model, so switching models re-encodes everything.

Keywords are matched using
[prompt ensembling](https://arxiv.org/abs/2103.00020) (Section 3.1.4):
each keyword is expanded into multiple prompt templates
(e.g. "a photo of a dog", "a photo of the dog"), embedded separately,
then averaged and renormalized. This improves zero-shot accuracy over
a single prompt.
