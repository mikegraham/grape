# grape

Find images matching keywords using [CLIP](https://openai.com/research/clip). Like grep, but for images.

```
grape 'golden retriever' ~/Photos/*.jpg
```

## Install

```
pip install -e .
```

Requires Python 3.10+. Model weights (~350 MB) are downloaded on first run.

## Usage

```
grape KEYWORDS PATH [PATH ...]
```

`KEYWORDS` is a comma-separated list (e.g. `dog`, `'cat,dog'`, `'golden retriever,tennis ball'`). Each `PATH` is an image file or a directory (with `-r`).

Images are ranked by average cosine similarity across all keywords. Image detection is content-based (`PIL.Image.verify`), not extension-based — binary files and non-images are skipped automatically.

### Options

| Flag | Description |
|------|-------------|
| `-r, --recursive` | Search directories recursively |
| `-t, --threshold SCORE` | Only show results with score >= SCORE (0.0-1.0) |
| `-n, --top N` | Show only top N results |
| `-s, --scores` | Show similarity scores alongside paths |
| `-v, --verbose` | Show per-keyword score breakdown (implies `-s`) |
| `-c, --count` | Print only the count of matching images (like `grep -c`) |
| `-q, --quiet` | Suppress progress and status messages on stderr |
| `--cache PATH` | Cache file for embeddings (created if it doesn't exist) |
| `--model MODEL` | CLIP model/pretrained tag (default: `ViT-B-32/laion2b_s34b_b79k`) |

### Examples

```bash
# Find dogs in a set of photos
grape dog *.jpg

# Multiple keywords — images are ranked by average similarity
grape 'cat,dog' *.jpg

# Search a directory recursively, show scores
grape -r -s sunset ~/Pictures

# Top 5 results above a threshold, verbose breakdown
grape -v -t 0.25 -n 5 'beach,palm tree' ~/Photos

# Count matching images (like grep -c)
grape -c -t 0.20 sunset ~/Pictures

# Use an embedding cache for fast repeat runs
grape --cache ~/.cache/grape.db dog ~/Photos/*.jpg
```

### Caching

Encoding images through CLIP is the bottleneck (~80 ms per image on CPU). The `--cache` flag stores image embeddings in a SQLite file so repeat runs skip re-encoding.

```bash
grape --cache my.db sunset ~/Photos/*.jpg   # first run: encodes all images
grape --cache my.db beach  ~/Photos/*.jpg   # second run: images already cached
```

Cache entries are keyed on absolute file path and model identifier. Entries are automatically invalidated when a file's size, mtime, or inode changes — no manual cache-busting needed.

### Models

The default model is `ViT-B-32/laion2b_s34b_b79k` — a good speed/quality tradeoff on CPU. Use `--model` to pick a different [open_clip](https://github.com/mlfoundations/open_clip) model:

```bash
grape --model ViT-L-14/laion2b_s32b_b82k sunset ~/Photos/*.jpg
```

The format is `model_name/pretrained_tag`. Run `python -c "import open_clip; print(open_clip.list_pretrained())"` to see available combinations.
