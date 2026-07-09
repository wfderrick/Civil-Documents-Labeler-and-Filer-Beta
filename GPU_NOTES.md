# GPU OCR settings

This project is configured for a single NVIDIA GPU.

Recommended settings in the web UI:

- OCR device: `auto` or `gpu`
- GPU ID: `0`
- DPI: `300`
- CPU workers: `1`
- CPU threads/worker: `4`

Why only one worker on GPU: multiple PaddleOCR worker processes usually fight over the same GPU memory. The app now forces one OCR engine when `OCR device = gpu`.

Verify your GPU install:

```powershell
python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.cuda.device_count())"
```

Expected:

```text
True
1
```

For CPU fallback, set OCR device to `cpu`, then try 3-4 workers and 4 threads per worker.
