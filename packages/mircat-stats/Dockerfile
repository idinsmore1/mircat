FROM ghcr.io/astral-sh/uv:bookworm-slim

# COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /bin/uv

# Update the OS
RUN apt-get update && \
    apt-get install -y clang gcc g++ python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --frozen --no-install-project --compile-bytecode

# Copy the project into the image
ADD . /app

# Sync the project
RUN uv sync --compile-bytecode

ENTRYPOINT ["uv", "run", "mircat-stats"]