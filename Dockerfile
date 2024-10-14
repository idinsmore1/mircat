FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.4.0 /uv /bin/uv

# Change the working directory to the `app` directory
RUN apt-get update && \
    apt-get install -y clang gcc g++ python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --compile-bytecode

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode

ENTRYPOINT ["uv", "run", "mircat"]