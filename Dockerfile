# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.11-slim

RUN pip install --no-cache-dir uv==0.8.13

WORKDIR /code

COPY ./pyproject.toml ./README.md ./uv.lock ./
COPY ./src ./src

# `--no-group ml` keeps PyTorch, CLAP, Whisper and FAISS out of this image.
# With them the layer reached ~8 GB and Agent Runtime never finished pulling it:
# the build succeeded, the container was never started and the create operation
# hung indefinitely. Retrieval now lives in Dockerfile.search-service.
RUN uv sync --frozen --no-dev --no-group eval --no-group ml

ARG COMMIT_SHA=""
ENV COMMIT_SHA=${COMMIT_SHA}

ARG AGENT_VERSION=0.0.0
ENV AGENT_VERSION=${AGENT_VERSION}

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "src.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
