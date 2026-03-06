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

Images are ranked by average cosine similarity across positive keywords. With `--exclude`, ranking becomes `include_mean - exclude_mean`. Image detection is content-based (`PIL.Image.verify`), not extension-based — binary files and non-images are skipped automatically.

### Options

| Flag | Description |
|------|-------------|
| `-r, --recursive` | Search directories recursively |
| `-t, --threshold SCORE` | Only show results with score >= SCORE (0.0-1.0) |
| `-n, --top N` | Show only top N results |
| `-x, --exclude KEYWORDS` | Comma-separated anti-match keywords |
| `--ensemble-prompts TEMPLATES` | Comma-separated prompt templates for ensembling (default: built-in set; each template must include `{}`) |
| `-s, --scores` | Show similarity scores alongside paths |
| `-v, --verbose` | Show per-keyword score breakdown (implies `-s`) |
| `-q, --quiet` | Suppress progress and status messages on stderr |
| `-print0` | Print matching paths as raw NUL-terminated records |
| `--view` | Open results in a native webview window using simple HTML with `<img>` tags |
| `--cache PATH` | Cache file for embeddings (created if it doesn't exist) |
| `--model MODEL` | CLIP model/pretrained tag (default: `ViT-B-16/laion2b_s34b_b88k`) |

By default, path output is shell-quoted (like `ls`) for safer copy/paste.

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

# Prefer "dog" while penalizing "cat"
grape -x cat dog ~/Pictures/*.jpg

# Use the built-in prompt ensemble (enabled by default)
grape dog ~/Pictures/*.jpg

# Override with custom prompt templates
grape --ensemble-prompts 'a photo of {},a sketch of {}' dog ~/Pictures/*.jpg

# Emit raw NUL-delimited paths for xargs -0
grape -print0 dog ~/Pictures/*.jpg | xargs -0 -n1 echo

# Open results in a native window
grape --view dog ~/Pictures/*.jpg

# Use an embedding cache for fast repeat runs
grape --cache ~/.cache/grape.db dog ~/Photos/*.jpg
```

`--view` is powered by `pywebview`.

### Caching

Encoding images through CLIP is the bottleneck (~80 ms per image on CPU). The `--cache` flag stores image embeddings in a SQLite file so repeat runs skip re-encoding.

```bash
grape --cache my.db sunset ~/Photos/*.jpg   # first run: encodes all images
grape --cache my.db beach  ~/Photos/*.jpg   # second run: images already cached
```

Cache entries are keyed on absolute file path and model identifier. Entries are automatically invalidated when a file's size, mtime, or inode changes — no manual cache-busting needed.

### Models

The default model is `ViT-B-16/laion2b_s34b_b88k` — a good speed/quality tradeoff on CPU. Use `--model` to pick a different [open_clip](https://github.com/mlfoundations/open_clip) model:

```bash
grape --model ViT-L-14/laion2b_s32b_b82k sunset ~/Photos/*.jpg
```

The format is `model_name/pretrained_tag`. Run `python -c "import open_clip; print(open_clip.list_pretrained())"` to see available combinations.
