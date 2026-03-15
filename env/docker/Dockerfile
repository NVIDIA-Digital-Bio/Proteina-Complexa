FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends rsync libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace/protein-foundation-models

# copy the repository files required for installation and building the environment
COPY pyproject.toml ./
COPY env/ ./env/
COPY community_models/ ./community_models/
# use uv installation script inside the docker
RUN bash /workspace/protein-foundation-models/env/build_uv_env.sh --root /workspace/
# copy the rest of the repo to support better layer cacheing (e.g. for development, where we change code but not dependencies)
COPY . .

# prevent CTRL-D to exit the shell
ENV IGNOREEOF=10

# ENV FOLDSEEK_EXEC="/workspace/.venv/bin/foldseek"
# ENV MMSEQS_EXEC="/workspace/.venv/bin/mmseqs"

# Install optional bioinformatics binaries if present in env/docker/internal/
# (sc, dssp, hbplus are not distributed with this repo — see README for details)
RUN for bin in sc dssp hbplus; do \
        src="env/docker/internal/${bin}"; \
        if [ -f "$src" ]; then \
            cp "$src" /usr/local/bin/ && chmod +x "/usr/local/bin/${bin}" && \
            echo "Installed ${bin}"; \
        else \
            echo "Skipping ${bin} (not found)"; \
        fi; \
    done
ENV SC_EXEC="/usr/local/bin/sc"
ENV DSSP_EXEC="/usr/local/bin/dssp"

# Set auxiliary models paths
ENV RF3_CKPT_PATH="/workspace/protein-foundation-models/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt"
ENV RF3_EXEC_PATH="/workspace/.venv/bin/rf3"
ENV AF2_DIR=/workspace/protein-foundation-models/community_models/ckpts/AF2

ENV PYTHONPATH="/workspace/protein-foundation-models/src"

# Default entrypoint: activate venv, run passed command or open interactive bash
COPY env/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
