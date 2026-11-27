# vLLM mamba overlap-copy race: #50729 evidence and build overlay

The pinned base image `intel/llm-scaler-vllm:0.26.0-b2`
(`sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`)
ships vLLM at commit `568afb3a1` (2026-07-27), **before** the upstream fix
[vllm-project/vllm#50729](https://github.com/vllm-project/vllm/pull/50729)
(merged 2026-08-17, commit `a02cfcc`). This document records the field evidence
for that gap and the minimal, pin-preserving overlay that `docker/Dockerfile`
applies at build time (`docker/patches/mamba-50729-barrier.patch`).

## The bug

`vllm/v1/worker/mamba_utils.py` copies mamba/SSM conv state through two Triton
paths: `batch_memcpy_kernel` and the speculative-decoding SD-conv tiled loop.
When a copy shifts data **left within the same physical block**
(`dst_ptr < src_ptr < dst_ptr + size`, i.e. overlapping ranges), the tiled
`tl.load`/`tl.store` loop has no memmove ordering: a tile's store can clobber
source bytes a later tile still needs to read. The conv state is corrupted
with no error raised and subsequent generation degrades.

Field symptom: intermittent tool-call arguments that are not valid JSON
on long agent sessions using the hybrid (attention + mamba)
Qwen3.8-27B EXL3 model (session `session-736f4800`, 2026-11-26 EST), plus
mid-turn output degradation. The corruption is probabilistic per state copy
(it only fires on overlapping left-shift copies), so short probes are not a
reliable repro and the failure was initially misread as a transport or
parsing issue.

## The upstream fix

[vllm-project/vllm#50729](https://github.com/vllm-project/vllm/pull/50729)
"[Bugfix][Mamba] Fix overlapping state copy race" (AndreasKaratzas),
**merged 2026-08-17** (commit `a02cfcc`):

- restructures the DS-conv copy for memmove-safe token ordering
  (`_memcpy_u64_tiled`);
- rewrites the SD-conv copy to the same helper;
- adds to `batch_memcpy_kernel`
  `is_left_overlap = dst_ptr < src_ptr and dst_ptr + size > src_ptr` and a
  conditional `tl.debug_barrier()` before the store.

## Why this overlay is a minimal adaptation

The full #50729 diff does not apply to the pinned base:

- its SD-conv hunk rewrites the copy to `_memcpy_u64_tiled`, which **does not
  exist** in vLLM `568afb3a1`; `patch -p1 --dry-run` on the base file: 3 of
  4 hunks apply, 1 rejected; a verbatim application would be a `NameError`;
- vLLM `568afb3a1` still carries the pre-#50729 inline tiled loop, which is
  exactly where the race lives.

`docker/patches/mamba-50729-barrier.patch` (3 hunks, ~1.4 KB) therefore ports
only the barrier half of #50729 to this base:

1. SD-conv inline tiled loop:
   `is_left_overlap = src_addr > dst_addr and src_addr < dst_addr + copy_size`
   plus `tl.debug_barrier()` before `tl.store`;
2. `batch_memcpy_kernel`:
   `is_left_overlap = dst_ptr < src_ptr and dst_ptr + size > src_ptr`
   plus `tl.debug_barrier()` before `tl.store`.

Scope: the default SD-conv layout and the generic `batch_memcpy_kernel`
fallback. The opt-in DS layout (`VLLM_SSM_CONV_STATE_LAYOUT=DS`) is only fully
covered by a base re-pin to vLLM `>= a02cfcc` (see below).

## Verification

All on the field lane (Intel Arc Pro B70, `xe` driver, podman,
`gpu_mem_util 0.94`, `max_model_len 262144`,
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
`--reasoning-parser qwen3`, `enable_thinking=false`), 2026-11-26/27 EST.

**Baseline (unpatched image `ghcr.io/0xsero/exl3xpu@21412bdd`):**

- `is_left_overlap` in the installed `mamba_utils.py`: 0;
- long tool-using sessions produced intermittent tool-call arguments that
  are not valid JSON (see above);
- an A/B canary on a second port reproduced mid-turn tool-call degradation on
  the unpatched image and not on the patched image.

**Overlay image (the file this patch produces):**

- `is_left_overlap` in the installed `mamba_utils.py`: 4 (fix marker);
- `python3 -m py_compile` clean; engine `/health` 200 in ~240 s (same load
  time as the unpatched image);
- end-to-end through the serving path:
  - plain chat completion: HTTP 200 in 0.38 s, `finish_reason=stop`;
  - forced `write` tool call: HTTP 200 in 0.74 s, `finish_reason=tool_calls`,
    arguments valid JSON: `{"path": "hello.txt", "content": "Hello from exl3."}`.

**Build-step proof (this overlay, executed inside the pinned base image):**

- `python3 -c "import vllm.v1.worker.mamba_utils; ..."` resolves
  `/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py`;
  `is_left_overlap` count before patching: 0;
- `patch` applies cleanly (GNU patch 2.7.6, present in the base image);
- the resulting file is **byte-identical** (sha256
  `fc2a6d6a4b98784ceea3c8cfe32bb411a8326d2b041034e6e468597ee9b52268`) to the
  field-verified file running in the production container;
- `py_compile` clean.

Notes on the Dockerfile `RUN` logic:

- path resolution pipes through `tail -1` because vLLM's import-time logging
  writes INFO/WARNING lines to **stdout** before the path is printed (observed
  with 0 active XPU devices);
- the `grep -q is_left_overlap || patch` guard makes the step idempotent
  (skipped when the base already carries the fix) and the final `grep` fails
  the build loudly if the barrier is absent afterwards;
- `tl.debug_barrier` exists in the base image's Triton 3.7.1.

## Recommended complete fix

Re-pin the base image to a vLLM `>= a02cfcc` build (or the next
`intel/llm-scaler-vllm` release that contains it). That brings the full
#50729 token-ordering fix, including the DS-conv path this overlay cannot
cover on the older base. The overlay is idempotent: once the re-pinned base
contains `is_left_overlap`, the Dockerfile step degrades to a no-op grep pass
and the patch file + step can be removed. A re-pin should go through the
project's usual XPU bit-exactness validation.

## Repro / trigger notes

- Host: Intel Arc Pro B70 (PCI `0000:67:00.0`), driver `xe`, ZES sysman
  enabled, container shm 32 g.
- Trigger: long agent sessions with repeated tool calls on a hybrid mamba
  model (spec-decode state shifts). Expect 0/1 or 1/1 on short probes, not a
  reliable per-request repro.
