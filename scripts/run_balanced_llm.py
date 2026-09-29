"""Step 6B: new sentiment + primary-emotion predictions, on the frozen balanced sample."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

EMOTIONS = ('ANGER', 'ANTICIPATION', 'DISGUST', 'FEAR', 'JOY', 'SADNESS', 'SURPRISE', 'TRUST')
RESPONSE_PATTERN = re.compile(r'SENTIMENT: (POSITIVE|NEUTRAL|NEGATIVE)\nEMOTION: (' + '|'.join(EMOTIONS) + r')')


def build_messages(prompt, review):
    """Allow only exact saved title and text into the model input."""
    return [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps({'title': review['title'], 'text': review['text']}, ensure_ascii=False)},
    ]


def parse_response(raw):
    """Full match only: preserve, never strip or repair, malformed responses."""
    match = RESPONSE_PATTERN.fullmatch(raw) if isinstance(raw, str) else None
    return match.groups() if match else (None, None)


def load_saved_reviews(source_dir):
    """Load only the frozen balanced sample; keep benchmarks separate from inputs."""
    source = source_dir / 'balanced_sample.json'
    frozen = json.loads(source.read_text(encoding='utf-8'))
    meta = json.loads((source_dir / 'metadata.json').read_text(encoding='utf-8'))
    for name in ('balanced_sample.json', 'balanced_sample.csv'):
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != meta['output_sha256'][name]:
            raise ValueError('Frozen sample fingerprint mismatch')
    if len(frozen) != 150 or [r['sample_id'] for r in frozen] != list(range(1,151)):
        raise ValueError('Expected exactly the frozen ordered 150 reviews')
    if len({r['source_line_number'] for r in frozen}) != 150:
        raise ValueError('Duplicate frozen sample identifiers')
    reviews = [{'review_number': r['sample_id'], 'title': r['review']['title'],
                'text': r['review']['text']} for r in frozen]
    benchmarks = {r['sample_id']: r['correct_sentiment'] for r in frozen}
    if any(sum(v == label for v in benchmarks.values()) != 50 for label in ('POSITIVE','NEUTRAL','NEGATIVE')):
        raise ValueError('Frozen saved benchmark counts changed')
    return reviews, benchmarks


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_batch(reviews, prompt, predict, output_dir, metadata, resume=False, benchmarks=None):
    """Keep a raw checkpoint after each model reply; never repair malformed text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'raw_results.jsonl'
    metadata_path = output_dir / 'run_metadata.json'
    results = []
    attempts_path = output_dir / 'request_attempts.jsonl'
    if resume:
        if json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Resume refused: prompt, model, or saved source changed')
        if checkpoint.exists():
            results = [json.loads(line) for line in checkpoint.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(results) > len(reviews):
            raise ValueError('Checkpoint exceeds the input batch')
        for row, review in zip(results, reviews):
            if any(row[key] != review[key] for key in ('review_number', 'title', 'text')):
                raise ValueError('Checkpoint review order or content changed')
            parsed = parse_response(row['raw_model_output'])
            if parsed != (row['new_sentiment'], row['primary_emotion']) or row['format_valid'] != (parsed[0] is not None):
                raise ValueError('Checkpoint response validation disagrees')
    else:
        if any(output_dir.iterdir()):
            raise FileExistsError('Use --resume or an empty output folder')
        write_json(metadata_path, metadata)

    attempts = [json.loads(line) for line in attempts_path.read_text(encoding='utf-8').splitlines()] if attempts_path.exists() else []
    if [a['review_number'] for a in attempts] != [r['review_number'] for r in results]:
        raise ValueError('Unresolved attempted request; refusing to resend it')

    with checkpoint.open('a', encoding='utf-8') as stream:
        for review in reviews[len(results):]:
            # Persist attempt BEFORE sending; a crash/timeout must never trigger a duplicate.
            with attempts_path.open('a', encoding='utf-8') as attempt_stream:
                attempt_stream.write(json.dumps({'review_number':review['review_number'], 'attempt':1}) + '\n')
                attempt_stream.flush()
                os.fsync(attempt_stream.fileno())
            raw = predict(build_messages(prompt, review))
            sentiment, emotion = parse_response(raw)
            result = {
                **review,
                'new_sentiment': sentiment,
                'primary_emotion': emotion,
                'raw_model_output': raw,
                'format_valid': sentiment is not None,
                'validation_error': None if sentiment is not None else 'Response does not match the exact two-line format',
                'prediction_received_utc': datetime.now(timezone.utc).isoformat(),
            }
            # Compare only after prediction; use the SAVED benchmark, never remap rating.
            correct = benchmarks[review['review_number']]
            result['correct_sentiment'] = correct
            result['is_correct'] = (sentiment == correct) if sentiment is not None else None
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            results.append(result)
            print(f"Processed {len(results)}/{len(reviews)}: {sentiment or 'INVALID'} / {emotion or 'INVALID'}", flush=True)

    invalid = [row for row in results if not row['format_valid']]
    summary = {
        'completed': len(results) == len(reviews) and all(row['raw_model_output'] is not None for row in results),
        'all_reviews_attempted': len(results) == len(reviews),
        'reviews_processed': len(results),
        'request_attempts': len(attempts_path.read_text(encoding='utf-8').splitlines()),
        'valid_format_count': len(results) - len(invalid),
        'invalid_format_count': sum(row['raw_model_output'] is not None for row in invalid),
        'missing_response_count': sum(row['raw_model_output'] is None for row in results),
        'emotion_counts': {emotion: sum(r['primary_emotion'] == emotion for r in results) for emotion in EMOTIONS},
        'new_sentiment_counts': {label: sum(r['new_sentiment'] == label for r in results) for label in ('POSITIVE', 'NEUTRAL', 'NEGATIVE')},
        'note': 'Separate three-class predictions compared with saved benchmarks only; no final analysis.',
    }
    write_json(output_dir / 'raw_results.json', results)
    write_json(output_dir / 'invalid_responses.json', invalid)
    write_json(output_dir / 'summary.json', summary)
    with (output_dir / 'raw_results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return summary


def main():
    # 1. Run with Hermes's existing packages; do not copy or print credentials.
    if importlib.util.find_spec('hermes_cli') is None:
        launcher = shutil.which('hermes')
        interpreter = Path(launcher).parent / 'python.exe' if launcher else None
        if interpreter and interpreter.exists() and interpreter.resolve() != Path(sys.executable).resolve():
            result = subprocess.run([str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]])
            raise SystemExit(result.returncode)
        raise RuntimeError('Run with the Python environment used by Hermes')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results_llm_emotion')
    args = parser.parse_args()
    # Restrict all output to this new Step 6B folder, never Step 1-6A files.
    if args.output_dir.resolve() == ROOT or not args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError('Output directory must be a child of the Step 6B directory')

    # 2. Read only the existing local reviews and the new prompt.
    source_dir = ROOT.parent / 'step6a/results'
    reviews, benchmarks = load_saved_reviews(source_dir)
    prompt_bytes = (ROOT / 'sentiment_emotion_prompt.txt').read_bytes()
    prompt = prompt_bytes.decode('utf-8')
    original_prompt = (ROOT.parent / 'step5a/sentiment_emotion_prompt.txt').read_text(encoding='utf-8')
    def emotion_section(value):
        return value.split('PRIMARY EMOTION\n', 1)[1].split('\nOUTPUT\n', 1)[0]
    if emotion_section(prompt) != emotion_section(original_prompt):
        raise ValueError('Existing eight-emotion rules must remain unchanged')

    # 3. Reuse the configured endpoint and existing authentication through Hermes.
    from hermes_cli.config import load_config
    from agent.auxiliary_client import resolve_provider_client
    settings = load_config()['model']
    client, model = resolve_provider_client(
        provider=settings['provider'], model=settings['default'],
        explicit_base_url=settings.get('base_url'),
    )
    if client is None:
        raise RuntimeError('No configured authenticated client is available')

    # The installed Codex adapter wraps an OpenAI SDK client with one streaming call.
    transport = getattr(client, '_real_client', client)
    if not hasattr(transport, 'max_retries'):
        raise RuntimeError('Cannot verify SDK retry policy')
    transport.max_retries = 0
    if transport.max_retries != 0:
        raise RuntimeError('SDK retries must be disabled')
    # Record only method/count, never headers, URLs, bodies, or credentials.
    wire_log = args.output_dir / 'http_model_requests.jsonl'
    def record_request(request):
        if request.method == 'POST':
            with wire_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'method':'POST'}) + '\n')
                stream.flush()
    transport._client.event_hooks.setdefault('request', []).append(record_request)

    def predict(messages):
        # Each request is independent and contains only instructions + title/text.
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=2048)
        return response.choices[0].message.content

    metadata = {
        'step': '6B', 'review_count': len(reviews),
        'source': 'step6a/results/balanced_sample.json',
        'source_csv_sha256': hashlib.sha256((source_dir / 'balanced_sample.csv').read_bytes()).hexdigest(),
        'source_json_sha256': hashlib.sha256((source_dir / 'balanced_sample.json').read_bytes()).hexdigest(),
        'prompt_file': 'sentiment_emotion_prompt.txt',
        'prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
        'original_prompt_sha256': hashlib.sha256((ROOT.parent / 'step5a/sentiment_emotion_prompt.txt').read_bytes()).hexdigest(),
        'provider': settings['provider'], 'model': model, 'max_tokens': 2048,
        'model_input_fields': ['title', 'text'],
        'sdk_max_retries': 0, 'application_retries': 0,
        'raw_output_scope': 'Content returned by Hermes adapter; Codex adapter strips outer whitespace upstream. This runner does not repair responses.',
        'response_validation': 'Exact full match; LF between lines; no trailing whitespace or additional text',
    }
    # 4. Store all new output separately, with raw responses and validation flags.
    summary = run_batch(reviews, prompt, predict, args.output_dir, metadata, args.resume, benchmarks=benchmarks)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Raw API exceptions can include sensitive request/transport details.
        print(f'Step 6B stopped ({type(error).__name__}). Check local paths and the Hermes connection. '
              'Completed responses and attempts are preserved; never resend an unresolved attempt.', file=sys.stderr)
        raise SystemExit(1)
