# exl3xpu: EXL3 inference on Intel Arc (Battlemage) via vLLM XPU.
FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
WORKDIR /opt/exl3xpu
COPY pyproject.toml ./
COPY exl3xpu ./exl3xpu
COPY csrc ./csrc
COPY scripts ./scripts
COPY models ./models
COPY bench ./bench
COPY tests ./tests
RUN bash scripts/build_ext.sh && pip install --no-deps --no-build-isolation -e . \
 && python3 -c "from importlib.metadata import entry_points as e; assert any(x.name=='exl3xpu' for x in e(group='vllm.general_plugins'))"
# vLLM mamba overlapping-state-copy race (fixed upstream by
# vllm-project/vllm#50729, merged 2026-08-17 as a02cfcc) is not in the pinned
# base: intel/llm-scaler-vllm:0.26.0-b2 ships vLLM 568afb3a1 (2026-07-27),
# before that merge. Left-shift mamba/conv-state copies overlap within the same
# physical block and the tiled load/store loop corrupts conv state (field
# symptom: tool-call arguments that are not valid JSON). This overlay applies
# the minimal pin-preserving barrier fix at build time
# (is_left_overlap + tl.debug_barrier, two kernels). It is idempotent: once
# the base is re-pinned to vLLM >= a02cfcc the grep passes and the patch step
# is skipped, and the overlay can be dropped. Evidence:
# docs/mamba-50729-evidence.md.
COPY docker/patches/mamba-50729-barrier.patch /tmp/mamba-50729-barrier.patch
RUN MAMBA="$(python3 -c 'import vllm.v1.worker.mamba_utils as m; print(m.__file__)' 2>/dev/null | tail -1)" \
    && test -f "$MAMBA" \
    && { grep -q is_left_overlap "$MAMBA" || patch "$MAMBA" < /tmp/mamba-50729-barrier.patch; } \
    && rm -f /tmp/mamba-50729-barrier.patch \
    && grep -q is_left_overlap "$MAMBA" \
    && echo "exl3xpu: mamba overlap-copy barrier (vllm-project/vllm#50729) present in $MAMBA"
ENV MODELS_DIR=/models
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
