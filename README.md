# grape

Find images matching keywords using [CLIP](https://openai.com/research/clip).
Like grep, but for images.

```
$ grape -R -s -k sunset ~/Pictures
0.327  ~/Pictures/vacation/beach_golden_hour.jpg
0.285  ~/Pictures/vacation/pier_evening.jpg
0.241  ~/Pictures/hiking/mountain_view.jpg
```

## Install

```
pip install .
```

Requires Python 3.10+.
Model weights (~350 MB for the default model) are downloaded on first run.

To use `--view` (opens results in a native window):

```
pip install '.[view]'
```

## Quick start

```bash
# search by keyword
grape -k sunset photo.jpg

# multiple keywords, ranked by average similarity
grape -k 'cat,dog' *.jpg

# recursive directory search
grape -R -k 'golden retriever' ~/Pictures

# find images similar to a reference image
grape --like reference.jpg -R ~/Pictures

# combine text and image queries
grape -k dog --like my_dog.jpg -R ~/Pictures
```

## Examples

```bash
# show scores (-s) or full per-keyword breakdown (-v)
grape -s -k sunset *.jpg
grape -v -k 'cat,dog,bird' *.jpg

# top 5 results above a similarity threshold
grape -n 5 -t 0.25 -k sunset -R ~/Pictures

# prefer "dog", penalize "cat" (score = include_mean - exclude_mean)
grape -k dog -x cat -R ~/Pictures

# multiple reference images
grape --like ref1.jpg --like ref2.jpg -R ~/Pictures

# pipe to other tools
grape -q -print0 -k cat -R ~/Pictures | xargs -0 cp -t ~/cats/

# browse results in a GUI window
grape --view -k sunset -R ~/Pictures

# cache embeddings for fast repeat queries
grape --cache grape.db -k sunset -R ~/Pictures
```

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
| `-v` | Per-keyword score breakdown (implies `-s`) |
| `-q` | Suppress status messages on stderr |
| `-print0` | NUL-separated output for `xargs -0` |
| `--view` | Browse results in a GUI window (requires `pip install '.[view]'`) |

### Configuration

| Flag | Description |
|------|-------------|
| `--cache PATH` | SQLite cache for image embeddings |
| `--model MODEL` | open_clip model/pretrained tag |

Image detection is content-based (`PIL.Image.verify`), not extension-based.
Output paths are shell-quoted by default.

## Scoring

Images are ranked by cosine similarity between CLIP embeddings.
Scores typically land in the 0.15--0.35 range for meaningful matches --
they are similarities, not probabilities, and even strong matches
rarely exceed 0.35.

With `-x`, the score becomes `mean(include) - mean(exclude)`.
With `--like`, reference image similarities are included in the mean
alongside text keywords.

Scores are **not comparable across models**.

## Caching

The first run without a cache encodes every image through the model
(~80 ms per image on CPU). With `--cache grape.db`, image embeddings
are stored in a SQLite file. Subsequent runs against the same images
are near-instant -- only new or changed files are re-encoded.

Cache entries are keyed by absolute path, file stat, and model
identifier. They auto-invalidate when a file's size, mtime, or inode
changes. Image detection results are cached too, so directory scans
get faster over time even for non-image files.

## Models

The default model is `ViT-B-16/laion2b_s34b_b88k`.
Use `--model` to pick any model from the
[open_clip pretrained registry](https://github.com/mlfoundations/open_clip):

```bash
grape --model ViT-L-14/laion2b_s32b_b82k -k sunset -R ~/Pictures
grape --model ViT-B-32/openai -k sunset -R ~/Pictures
```

The format is `model_name/pretrained_tag`.
Larger models are more accurate but slower.
