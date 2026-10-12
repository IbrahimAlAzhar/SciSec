#!/usr/bin/env python3
"""Own the single-GPU vLLM process group, health checks, logs, and cleanup."""
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def stop_group(proc):
    # The group can still contain workers after its leader exits.
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    proc.wait()


def parser():
    # This wrapper forwards --mode to run_baseline.py. Prefix matching would
    # otherwise consume it as --model and replace the checkpoint with 'replay'.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--model', default='path/to/models/gemma3_27b_it')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-model-len', type=int, default=16384)
    return p


def main():
    a, rest = parser().parse_known_args()
    out = Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'output.txt').exists(): raise SystemExit('Refusing to overwrite existing run')
    def log(message):
        print(message, flush=True)
        with (out/'output.txt').open('a') as f: f.write(message+'\n')
    code = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1')
    a.model = str(Path(a.model).expanduser().resolve())
    log(f'Model directory: {a.model}')
    log(f'Forwarded evaluation arguments: {rest}')
    for f in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        path = Path(a.model)/f
        if not path.is_file():
            message = f'Missing or inaccessible model file: {path}'
            log(message)
            raise SystemExit(message)
    # Cap vLLM's budget at about 35 GiB even if scheduled on a larger GPU.
    gpu = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    if not gpu or gpu == '-1': raise SystemExit('No allocated CUDA GPU')
    total = float(subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
    util = min(0.85, 35840 / total)
    if total < 38000: raise SystemExit(f'Expected a 40 GB GPU; detected {total} MiB')
    log(f'GPU total={total} MiB; vLLM utilization={util:.4f}; 4-bit weights; one sequence; eager mode')
    log('Requested memory budget is not a measured guarantee. Inspect gpu_memory.csv for actual use.')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    log(f'Starting local vLLM port={port}, model={a.model}, max_model_len={a.max_model_len}')
    (out/'server.port').write_text(str(port))
    proc = None; monitor = None; runner = None
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out/'vllm.log').open('w') as server_log, (out/'gpu_memory.csv').open('w') as gpu_log:
            proc = subprocess.Popen(['bash', str(code/'serve_vllm.sh'), a.model, str(port),
                str(a.max_model_len), str(util)], env=env, stdout=server_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            monitor = subprocess.Popen(['nvidia-smi', '-i', gpu,
                '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
                '--format=csv', '--loop=10'], stdout=gpu_log, stderr=subprocess.DEVNULL)
            import httpx
            with httpx.Client(trust_env=False, timeout=10,
                 headers={'Authorization': 'Bearer '+(env.get('VLLM_API_KEY') or 'EMPTY')}) as client:
                deadline = time.monotonic()+1800
                while True:
                    if proc.poll() is not None: raise RuntimeError(f'vLLM exited: {proc.returncode}')
                    if time.monotonic() >= deadline: raise TimeoutError('vLLM readiness exceeded 30 minutes')
                    try:
                        health = client.get(f'http://127.0.0.1:{port}/health')
                        models = client.get(f'http://127.0.0.1:{port}/v1/models')
                        if models.status_code in (401, 403): raise RuntimeError('API key mismatch')
                        if health.status_code == 200 and models.status_code == 200:
                            if a.model not in [x['id'] for x in models.json()['data']]:
                                raise RuntimeError('Unexpected model on selected port')
                            break
                    except (httpx.ConnectError, httpx.TimeoutException): pass
                    time.sleep(5)
            log('Server ready. Full model logs: vllm.log; measurements: gpu_memory.csv')
            runner = subprocess.Popen([sys.executable, str(code/'run_baseline.py'),
                '--model', a.model, '--url', f'http://127.0.0.1:{port}/v1',
                '--out-dir', str(out), '--max-model-len', str(a.max_model_len), *rest], env=env,
                start_new_session=True)
            rc = runner.wait()
            log(f'Baseline exit code={rc}; nonzero means errors require inspection')
            return rc
    except BaseException as exc:
        log(f'Job failed: {type(exc).__name__}: {exc}')
        if (out/'vllm.log').exists():
            log('\n'.join((out/'vllm.log').read_text(errors='replace').splitlines()[-50:]))
        raise
    finally:
        if runner is not None: stop_group(runner)
        if proc is not None: stop_group(proc)
        if monitor is not None:
            monitor.terminate()
            try: monitor.wait(timeout=5)
            except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()


if __name__ == '__main__': raise SystemExit(main())
