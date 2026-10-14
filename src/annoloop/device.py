"""Read-only NVIDIA/CUDA diagnostics; never fall back to CPU detection."""
import json
import subprocess


def gpu_idle():
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Không kiểm tra được NVIDIA GPU") from error
    if result.returncode:
        raise RuntimeError("NVIDIA driver/GPU không sẵn sàng")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("Driver không trả danh sách compute PID hợp lệ; chưa xác minh GPU idle")
    pids = lines
    if pids:
        raise RuntimeError("GPU đang có process: " + ", ".join(pids))


def probe(python):
    command = [str(python), "-c", "import json,torch; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available(),devices=torch.cuda.device_count())))"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            return dict(ready=False, reason="venv_cuda_import_failed", detail=result.stderr[-2000:])
        status = json.loads(result.stdout.splitlines()[-1])
        return dict(ready=status["available"], **status)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return dict(ready=False, reason=type(error).__name__)
