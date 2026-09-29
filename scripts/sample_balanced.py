"""Reproducible full-stream stratified reservoir sample; standard library only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DATASET_URL = ('https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
               'raw/review_categories/Gift_Cards.jsonl.gz')

SEED = 20260929
GROUPS = ('1-2', '3', '4-5')
LABELS = {'1-2': 'NEGATIVE', '3': 'NEUTRAL', '4-5': 'POSITIVE'}


def sample_lines(lines, seed=SEED, size=50):
    """Algorithm R, one shared seeded RNG, three rating-only reservoirs.

    Read to EOF. The kth eligible review in a group replaces a reservoir
    slot with probability size/k. Keep only size reviews per group.
    No correct_sentiment label is attached during this selection phase.
    """
    rng = random.Random(seed)
    pools = {group: [] for group in GROUPS}
    seen = dict.fromkeys(GROUPS, 0)
    stars = dict.fromkeys(range(1, 6), 0)
    count = skipped = 0
    stream_hash = hashlib.sha256()
    for line_number, raw in enumerate(lines, 1):
        stream_hash.update(raw)
        if not raw.strip():
            continue
        review = json.loads(raw)
        if not isinstance(review, dict):
            raise ValueError(f'Non-object review on line {line_number}')
        count += 1
        rating = review.get('rating')
        if type(rating) not in (int, float) or rating not in stars:
            skipped += 1
            continue
        stars[rating] += 1
        group = '1-2' if rating <= 2 else ('3' if rating == 3 else '4-5')
        seen[group] += 1
        # Retain hashes for selected source records without normalizing their content.
        item = {'source_line_number': line_number,
                'source_record_sha256': hashlib.sha256(raw).hexdigest(),
                'review': review}
        if len(pools[group]) < size:
            pools[group].append(item)
        else:
            slot = rng.randrange(seen[group])
            if slot < size:
                pools[group][slot] = item
    if any(len(pool) != size for pool in pools.values()):
        raise ValueError(f'Not enough reviews in each rating group: {seen}')
    selected = sorted([row for pool in pools.values() for row in pool],
                      key=lambda row: row['source_line_number'])
    return selected, {'review_count': count, 'eligible_rating_group_counts': seen,
                      'star_rating_counts': stars, 'invalid_rating_reviews_skipped': skipped,
                      'uncompressed_stream_sha256': stream_hash.hexdigest()}


def add_labels(selected):
    """Only after sampling has finished, derive the benchmark labels."""
    result = []
    for row in selected:
        rating = row['review']['rating']
        label = 'NEGATIVE' if rating <= 2 else ('NEUTRAL' if rating == 3 else 'POSITIVE')
        result.append(dict(row, correct_sentiment=label))
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_sample(sample):
    """Require 150 distinct source records and distinct full review contents."""
    if len(sample) != 150:
        raise ValueError('Sample must contain exactly 150 reviews')
    lines = {r['source_line_number'] for r in sample}
    records = {r['source_record_sha256'] for r in sample}
    contents = {json.dumps(r['review'], sort_keys=True, ensure_ascii=False) for r in sample}
    identities = {(r['review'].get('user_id'), r['review'].get('parent_asin'),
                   r['review'].get('timestamp')) for r in sample}
    if any(len(keys) != 150 for keys in (lines, records, contents, identities)):
        raise ValueError('Duplicate source record, review content, or user/product/timestamp identity')
    counts = dict(Counter(r['correct_sentiment'] for r in sample))
    if counts != {'POSITIVE': 50, 'NEUTRAL': 50, 'NEGATIVE': 50}:
        raise ValueError(f'Incorrect class counts: {counts}')
    for row in sample:
        rating = row['review']['rating']
        expected = 'POSITIVE' if rating in (4, 5) else ('NEUTRAL' if rating == 3 else 'NEGATIVE')
        if row['correct_sentiment'] != expected:
            raise ValueError('Rating and derived label disagree')
    return counts


def run(source, output, seed=SEED):
    if output.exists():
        raise FileExistsError('Choose a new output folder; prior results are immutable')
    # Streaming gzip reading reaches EOF, including gzip integrity checks.
    with gzip.open(source, 'rb') as stream:
        selected, population = sample_lines(stream, seed=seed, size=50)
    sample = add_labels(selected)
    counts = validate_sample(sample)
    for number, row in enumerate(sample, 1):
        row['sample_id'] = number
    metadata = {
        'seed': seed, 'python_version': platform.python_version(),
        'sampling_method': 'Stratified reservoir sampling, Algorithm R, size 50 per rating group; one shared random.Random(seed) generator; rng.randrange(group_seen) called only after a group reservoir is full.',
        'selection_order': 'Entire gzip stream in original line order; final selected rows sorted by source line number.',
        'label_timing': 'Ratings used only for grouping during selection; correct_sentiment attached after EOF and selection completion.',
        'rating_mapping': {'4-5': 'POSITIVE', '3': 'NEUTRAL', '1-2': 'NEGATIVE'},
        'dataset_url': DATASET_URL, 'local_source': str(source.resolve()),
        'dataset_compressed_sha256': digest(source),
        'script_sha256': digest(Path(__file__)),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sample_size': len(sample), 'class_counts': counts,
        'uniqueness_checks': ['source line', 'source line SHA-256', 'canonical complete review', 'user_id + parent_asin + timestamp'],
        'filtering': 'Only nonnumeric or out-of-range ratings excluded; no filtering by title/text, sentiment, or earlier sample membership. Malformed JSON fails. Duplicate selected reviews fail validation rather than being silently replaced.',
        'population': population, 'model_calls': 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'balanced_sample.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['sample_id', 'source_line_number', 'source_record_sha256', 'title', 'text',
              'star_rating', 'correct_sentiment', 'user_id', 'asin', 'parent_asin',
              'timestamp', 'verified_purchase', 'helpful_vote', 'images']
    with (output / 'balanced_sample.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sample:
            review = row['review']
            flat = {key: review.get(key) for key in fields}
            for key in ('sample_id', 'source_line_number', 'source_record_sha256', 'correct_sentiment'):
                flat[key] = row[key]
            flat['star_rating'] = review['rating']
            flat['images'] = json.dumps(review.get('images'), ensure_ascii=False)
            writer.writerow(flat)
    metadata['output_sha256'] = {name: digest(output / name) for name in ('balanced_sample.json', 'balanced_sample.csv')}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/Gift_Cards.jsonl.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Output folder exists; select a new --output-dir')
    # Download once, preserving the compressed source for fully local reruns.
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        partial = args.source.with_suffix(args.source.suffix + '.part')
        with urlopen(DATASET_URL, timeout=120) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        partial.replace(args.source)
    print(json.dumps(run(args.source, args.output_dir, args.seed), indent=2))
