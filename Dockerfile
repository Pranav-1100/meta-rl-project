# PhonePilot environment container.
# Builds on OpenEnv's official base image, installs the project via uv, and serves
# the FastAPI app on port 8000. Designed to work both locally and on Hugging Face Spaces.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available for any VCS-pinned dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy the whole project tree.
COPY . /app

# Ensure uv is on PATH (base image may not ship it).
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install the project + its deps into a .venv. Uses uv.lock if present.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# --- runtime stage ---
FROM ${BASE_IMAGE}
WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "phonepilot_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
