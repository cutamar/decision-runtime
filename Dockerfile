FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/decision-lab

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

# The default image includes all three model choices. Set ENABLE_ADAPT=0 for
# a smaller image that supports sparse and frozen MiniLM candidates.
ARG ENABLE_ADAPT=1
RUN if [ "$ENABLE_ADAPT" = "1" ]; then \
        python -m pip install --no-cache-dir 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu \
        && python -m pip install --no-cache-dir '.[lab,adapt]'; \
    elif [ "$ENABLE_ADAPT" = "0" ]; then \
        python -m pip install --no-cache-dir '.[lab]'; \
    else \
        echo 'ENABLE_ADAPT must be 0 or 1' >&2; exit 2; \
    fi

RUN useradd --create-home --uid 10001 lab \
    && mkdir /data \
    && chown lab:lab /data
USER lab

EXPOSE 8765
CMD ["decision-web", "--host", "0.0.0.0", "--port", "8765", \
     "--workdir", "/data/web-runs", \
     "--source", "/data/models/all-MiniLM-L6-v2-bc57282"]
