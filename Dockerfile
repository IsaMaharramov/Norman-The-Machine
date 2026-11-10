FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY cpp ./cpp
COPY src ./src
RUN pip install --no-cache-dir ".[test]"

COPY tests ./tests
COPY examples ./examples
COPY docs ./docs

ENV MPLBACKEND=Agg
RUN mkdir -p /app/out

CMD ["norman", "evaluate", "--controller", "pid"]
