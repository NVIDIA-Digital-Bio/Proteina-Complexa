#!/bin/bash
# ==============================================================================
# Docker Operations Script
# ==============================================================================
# Usage: ./docker-ops.sh [command] [options]
#
# Commands:
#   pull                          Pull container from registry
#   build [--clean] [--base IMG]  Build container locally
#   push [TAG]                    Push container to registry
#   squash [-o FILE]              Create squashfs for Slurm (requires enroot)
#   run [options] [CMD]           Run container locally
#   attach [CMD]                  Attach to running container
#   stop                          Stop running container
#
# Configuration is read from .env file (override path with PROTEINA_ENV).
# If .env is missing, .env_example is copied as a starting point.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# Environment Loading
# ==============================================================================

load_env() {
    local env_path="${PROTEINA_ENV:-.env}"
    # Make relative paths relative to PROJECT_DIR
    if [[ "$env_path" != /* ]]; then
        env_path="$PROJECT_DIR/$env_path"
    fi
    ENV_FILE="$env_path"

    if [[ ! -f "$ENV_FILE" ]]; then
        local example="$PROJECT_DIR/.env_example"
        if [[ -f "$example" ]]; then
            echo "Environment file not found: $ENV_FILE"
            echo "Copying from $example -> $ENV_FILE"
            cp "$example" "$ENV_FILE"
            echo ""
            echo "A default .env file has been created. Please edit it with your settings:"
            echo "  $ENV_FILE"
            echo ""
            echo "Then re-run this script."
            exit 1
        else
            echo "Error: Neither $ENV_FILE nor $example found."
            echo "Cannot proceed without configuration."
            exit 1
        fi
    fi

    echo "Loading environment from: $ENV_FILE"
    source "$ENV_FILE"
}

require_vars() {
    local missing=()
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            missing+=("$var")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo ""
        echo "Error: The following required variables are not set in $ENV_FILE:"
        for var in "${missing[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set them in $ENV_FILE and re-run."
        exit 1
    fi
}

# ==============================================================================
# GPU Detection
# ==============================================================================

detect_gpu() {
    GPU_RUNTIME=""

    if ! command -v docker &>/dev/null; then
        echo "GPU detection: docker not found in PATH"
        return
    fi

    # Check if NVIDIA GPU hardware is available
    if ! command -v nvidia-smi &>/dev/null && [[ ! -e /dev/nvidia0 ]]; then
        echo "GPU detection: no NVIDIA GPU detected, running without GPU support"
        return
    fi

    local docker_version
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
    local min_version="19.03.0"

    if [[ "$min_version" == "$(echo -e "$docker_version\n$min_version" | sort -V | head -1)" ]]; then
        GPU_RUNTIME="--gpus all"
        echo "GPU detection: using --gpus all (Docker $docker_version)"
    else
        GPU_RUNTIME="--runtime=nvidia"
        echo "GPU detection: using --runtime=nvidia (Docker $docker_version)"
    fi
}

# ==============================================================================
# Docker Login
# ==============================================================================

docker_login() {
    require_vars REGISTRY REGISTRY_USER

    if [[ -n "${GITLAB_TOKEN:-}" && "${GITLAB_TOKEN}" != "TOKEN_HERE" ]]; then
        echo "Logging into registry: $REGISTRY"
        docker login "$REGISTRY" -u "$REGISTRY_USER" -p "$GITLAB_TOKEN"
    else
        echo "Warning: GITLAB_TOKEN is not set or is a placeholder. Skipping docker login."
        echo "  Push/pull to private registries may fail."
    fi
}

# ==============================================================================
# Commands
# ==============================================================================

cmd_pull() {
    require_vars DOCKER_IMAGE REGISTRY REGISTRY_USER

    echo ""
    echo "=== Pull ==="
    docker_login
    echo "Pulling image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE"
    echo "Done: pulled $DOCKER_IMAGE"
}

cmd_build() {
    require_vars DOCKER_IMAGE DOCKERFILE_PATH

    local clean=""
    local base_image=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean) clean="--no-cache"; shift ;;
            --base) base_image="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; echo "Usage: docker-ops.sh build [--clean] [--base IMAGE]"; exit 1 ;;
        esac
    done

    local dockerfile="$DOCKERFILE_PATH"
    # Make relative paths relative to PROJECT_DIR
    if [[ "$dockerfile" != /* ]]; then
        dockerfile="$PROJECT_DIR/$dockerfile"
    fi

    if [[ ! -f "$dockerfile" ]]; then
        echo "Error: Dockerfile not found: $dockerfile"
        exit 1
    fi

    local git_hash="unknown"
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git_hash=$(git rev-parse --short HEAD)
    fi

    echo ""
    echo "=== Build ==="
    echo "Image:      $DOCKER_IMAGE"
    echo "Dockerfile: $dockerfile"
    echo "Context:    $PROJECT_DIR"
    echo "Git hash:   $git_hash"
    [[ -n "$clean" ]] && echo "Cache:      disabled (--clean)"
    [[ -n "$base_image" ]] && echo "Base image: $base_image"
    echo ""

    local -a build_cmd=(docker build --network host)
    build_cmd+=(-t "$DOCKER_IMAGE")
    build_cmd+=(--label "com.nvidia.proteinfoundation.git_hash=$git_hash")
    build_cmd+=(-f "$dockerfile")

    [[ -n "$clean" ]] && build_cmd+=("$clean")
    [[ -n "$base_image" ]] && build_cmd+=(--build-arg "BASE_IMAGE=$base_image")

    echo "Running: DOCKER_BUILDKIT=1 ${build_cmd[*]} $PROJECT_DIR"
    echo ""
    DOCKER_BUILDKIT=1 "${build_cmd[@]}" "$PROJECT_DIR"

    echo ""
    echo "Done: built $DOCKER_IMAGE"
}

cmd_push() {
    require_vars DOCKER_IMAGE REGISTRY REGISTRY_USER

    local tag="${1:-}"
    local img_base="${DOCKER_IMAGE%%:*}"

    echo ""
    echo "=== Push ==="
    docker_login

    local push_image
    if [[ -n "$tag" ]]; then
        push_image="$img_base:$tag"
        echo "Tagging: $DOCKER_IMAGE -> $push_image"
        docker tag "$DOCKER_IMAGE" "$push_image"
    else
        push_image="$DOCKER_IMAGE"
    fi

    echo "Pushing: $push_image"
    docker push "$push_image"
    echo "Done: pushed $push_image"
}

cmd_squash() {
    # -------------------------------------------------------------------------
    # Create a squashfs (.sqsh) file from a Docker image using enroot.
    #
    # This must be executed on a SLURM cluster (not a login node — use a
    # compute node to avoid OOM).
    #
    # Prerequisites:
    #   1. enroot must be installed and in PATH.
    #   2. Configure registry credentials in ~/.config/enroot/.credentials:
    #        machine <your-registry> login <user> password <token>
    #   3. Disable curl proxy by adding to ~/.curlrc:
    #        noproxy = *
    #
    # Example (manual):
    #   srun -A <account> -p <partition> -N1 -n1 --pty bash
    #   enroot import -o protein-foundation-models.sqsh \
    #       docker://<your-registry>:5005#<org>/protein-foundation-models:<tag>
    #
    # Usage:
    #   docker-ops.sh squash              # output: <repo>.<tag>.sqsh
    #   docker-ops.sh squash -o out.sqsh  # custom output path
    # -------------------------------------------------------------------------
    require_vars DOCKER_IMAGE

    command -v enroot >/dev/null 2>&1 || {
        echo "Error: enroot not found in PATH."
        echo "Squash requires enroot. Install it or run on a SLURM cluster node."
        exit 1
    }

    local custom_output=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output) custom_output="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; echo "Usage: docker-ops.sh squash [-o OUTPUT_PATH]"; exit 1 ;;
        esac
    done

    # Parse image into enroot URI: registry#path:tag -> docker://registry#path:tag
    # DOCKER_IMAGE format: <registry>:5005/<org>/protein-foundation-models:<tag>
    # enroot format:       docker://<registry>:5005#<org>/protein-foundation-models:<tag>
    local registry="${DOCKER_IMAGE%%/*}"
    local path_and_tag="${DOCKER_IMAGE#*/}"
    local enroot_uri="docker://${registry}#${path_and_tag}"

    # Determine output filename
    local out
    if [[ -n "$custom_output" ]]; then
        out="$custom_output"
    else
        local img_last="${DOCKER_IMAGE##*/}"
        local repo="${img_last%%:*}"
        local tag="${img_last##*:}"
        [[ "$repo" == "$tag" ]] && tag="latest"
        out="${repo}.${tag}.sqsh"
    fi

    echo ""
    echo "=== Squash ==="
    echo "Image:      $DOCKER_IMAGE"
    echo "Enroot URI: $enroot_uri"
    echo "Output:     $out"
    echo ""

    echo "Importing image with enroot..."
    set -x
    enroot import -o "$out" "$enroot_uri"
    set +x

    echo ""
    echo "Done: $out"
}

cmd_run() {
    require_vars DOCKER_IMAGE CONTAINER_NAME DOCKER_REPO_PATH

    local mount_code=1
    local mount_ssh=1
    local mount_claude=1
    local -a cmd=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-code-mount) mount_code=0; shift ;;
            --no-ssh) mount_ssh=0; shift ;;
            --no-claude) mount_claude=0; shift ;;
            --) shift; cmd=("$@"); break ;;
            -*) echo "Unknown option: $1"; echo "Usage: docker-ops.sh run [--no-code-mount] [--no-ssh] [--no-claude] [--] [CMD...]"; exit 1 ;;
            *) cmd=("$@"); break ;;
        esac
    done

    [[ ${#cmd[@]} -eq 0 ]] && cmd=("bash")

    echo ""
    echo "=== Run ==="
    echo "Image:     $DOCKER_IMAGE"
    echo "Container: $CONTAINER_NAME"
    echo "Command:   ${cmd[*]}"
    echo ""

    # GPU detection
    detect_gpu

    # Build docker run command using array for safe quoting
    local -a docker_cmd=(docker run --rm -it --network host)
    docker_cmd+=(--name "$CONTAINER_NAME")
    docker_cmd+=(--shm-size=4g)
    docker_cmd+=(--ulimit memlock=-1)
    docker_cmd+=(--ulimit stack=67108864)

    # GPU
    if [[ -n "${GPU_RUNTIME:-}" ]]; then
        # GPU_RUNTIME may contain multiple words (e.g. "--gpus all")
        read -ra gpu_args <<< "$GPU_RUNTIME"
        docker_cmd+=("${gpu_args[@]}")
    fi

    # Environment variables
    docker_cmd+=(-e "PROJECT_HOME=$DOCKER_REPO_PATH")
    docker_cmd+=(-e "HOME=$DOCKER_REPO_PATH")
    if [[ -n "${DOCKER_PYTHONPATH:-}" ]]; then
        docker_cmd+=(-e "PYTHONPATH=$DOCKER_PYTHONPATH")
    fi

    if [[ -n "${DOCKER_DATA_PATH:-}" ]]; then
        docker_cmd+=(-e "DATA_PATH=$DOCKER_DATA_PATH")
    fi

    # Weights & Biases
    if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != "YOUR WANDB KEY" ]]; then
        docker_cmd+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
    fi
    if [[ -n "${WANDB_ENTITY:-}" && "${WANDB_ENTITY}" != "YOUR WANDB ENTITY" ]]; then
        docker_cmd+=(-e "WANDB_ENTITY=$WANDB_ENTITY")
    fi

    # Map LOCAL_CODE_PATH to the container repo path so that dotenv
    # expansions (COMMUNITY_MODELS_PATH, ESM_DIR, etc.) resolve correctly.
    docker_cmd+=(-e "LOCAL_CODE_PATH=$DOCKER_REPO_PATH")

    # Logging / model config
    if [[ -n "${LOGURU_LEVEL:-}" ]]; then
        docker_cmd+=(-e "LOGURU_LEVEL=$LOGURU_LEVEL")
    fi
    if [[ -n "${USE_V2_COMPLEXA_ARCH:-}" ]]; then
        docker_cmd+=(-e "USE_V2_COMPLEXA_ARCH=$USE_V2_COMPLEXA_ARCH")
    fi

    # Cache / HuggingFace environment variables
    if [[ -n "${DOCKER_CACHE_DIR:-}" ]]; then
        docker_cmd+=(-e "CACHE_DIR=$DOCKER_CACHE_DIR")
    fi
    if [[ -n "${DOCKER_HF_HOME:-}" ]]; then
        docker_cmd+=(-e "HF_HOME=$DOCKER_HF_HOME")
    fi
    if [[ -n "${DOCKER_HF_HUB_CACHE:-}" ]]; then
        docker_cmd+=(-e "HF_HUB_CACHE=$DOCKER_HF_HUB_CACHE")
    fi

    # Code mount
    if [[ $mount_code -eq 1 ]]; then
        echo "Mounting code: $PROJECT_DIR -> $DOCKER_REPO_PATH"
        docker_cmd+=(-v "$PROJECT_DIR:$DOCKER_REPO_PATH" -w "$DOCKER_REPO_PATH")
    else
        echo "Code mount: disabled (--no-code-mount)"
    fi

    # Data mount
    if [[ -n "${LOCAL_DATA_PATH:-}" && -n "${DOCKER_DATA_PATH:-}" ]]; then
        echo "Mounting data: $LOCAL_DATA_PATH -> $DOCKER_DATA_PATH"
        docker_cmd+=(-v "$LOCAL_DATA_PATH:$DOCKER_DATA_PATH")
    else
        echo "Data mount: skipped (LOCAL_DATA_PATH or DOCKER_DATA_PATH not set)"
    fi

    # Cache mount
    if [[ -n "${LOCAL_CACHE_DIR:-}" && -n "${DOCKER_CACHE_DIR:-}" ]]; then
        echo "Mounting cache: $LOCAL_CACHE_DIR -> $DOCKER_CACHE_DIR"
        docker_cmd+=(-v "$LOCAL_CACHE_DIR:$DOCKER_CACHE_DIR")
    else
        echo "Cache mount: skipped (LOCAL_CACHE_DIR or DOCKER_CACHE_DIR not set)"
    fi

    # Checkpoint mount
    if [[ -n "${LOCAL_CHECKPOINT_PATH:-}" && -n "${DOCKER_CHECKPOINT_PATH:-}" ]]; then
        echo "Mounting checkpoints: $LOCAL_CHECKPOINT_PATH -> $DOCKER_CHECKPOINT_PATH"
        docker_cmd+=(-v "$LOCAL_CHECKPOINT_PATH:$DOCKER_CHECKPOINT_PATH")
        # Set CKPT_PATH so YAML configs using ${oc.env:CKPT_PATH} resolve inside container
        docker_cmd+=(-e "CKPT_PATH=$DOCKER_CHECKPOINT_PATH")
    else
        echo "Checkpoint mount: skipped (LOCAL_CHECKPOINT_PATH or DOCKER_CHECKPOINT_PATH not set)"
    fi

    # SSH mount — target must match container HOME for git/ssh to find keys
    if [[ $mount_ssh -eq 1 && -d "$HOME/.ssh" ]]; then
        echo "Mounting SSH: $HOME/.ssh -> $DOCKER_REPO_PATH/.ssh (read-only)"
        docker_cmd+=(-v "$HOME/.ssh:$DOCKER_REPO_PATH/.ssh:ro")
    elif [[ $mount_ssh -eq 0 ]]; then
        echo "SSH mount: disabled (--no-ssh)"
    else
        echo "SSH mount: skipped (~/.ssh not found)"
    fi

    # CLUSTER_SSH_KEY mount — mount the specific key file so slurm_helper.sh can find it
    if [[ -n "${CLUSTER_SSH_KEY:-}" ]]; then
        local host_key="${CLUSTER_SSH_KEY/#\~/$HOME}"
        if [[ -f "$host_key" ]]; then
            local key_name
            key_name="$(basename "$host_key")"
            local container_key="$DOCKER_REPO_PATH/.ssh/$key_name"
            echo "Mounting SSH key: $host_key -> $container_key (read-only)"
            docker_cmd+=(-v "$host_key:$container_key:ro")
            docker_cmd+=(-e "CLUSTER_SSH_KEY=$container_key")
        else
            echo "Warning: CLUSTER_SSH_KEY file not found: $host_key"
        fi
    fi

    # Claude binary mount
    if [[ $mount_claude -eq 1 ]]; then
        local claude_path
        claude_path="$(command -v claude 2>/dev/null || true)"
        if [[ -n "$claude_path" ]]; then
            echo "Mounting Claude: $claude_path -> /usr/local/bin/claude (read-only)"
            docker_cmd+=(-v "$claude_path:/usr/local/bin/claude:ro")
        else
            echo "Claude mount: skipped (claude binary not found in PATH)"
        fi
    else
        echo "Claude mount: disabled (--no-claude)"
    fi

    # Custom mounts from DOCKER_MOUNTS (comma-separated "src:dst" pairs)
    if [[ -n "${DOCKER_MOUNTS:-}" ]]; then
        IFS=',' read -ra mounts <<< "$DOCKER_MOUNTS"
        for mount in "${mounts[@]}"; do
            mount="$(echo "$mount" | xargs)"  # trim whitespace
            if [[ -n "$mount" ]]; then
                echo "Mounting custom: $mount"
                docker_cmd+=(-v "$mount")
            fi
        done
    fi

    echo ""
    echo "Starting container..."
    set -x
    "${docker_cmd[@]}" "$DOCKER_IMAGE" "${cmd[@]}"
    set +x
}

cmd_attach() {
    require_vars CONTAINER_NAME

    echo ""
    echo "=== Attach ==="
    echo "Container: $CONTAINER_NAME"

    # Find container ID by name
    local container_id
    container_id=$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}' | head -1)

    if [[ -z "$container_id" ]]; then
        echo "Error: No running container found with name '$CONTAINER_NAME'"
        echo ""
        echo "Running containers:"
        docker ps --format '  {{.Names}}\t{{.Image}}\t{{.Status}}'
        exit 1
    fi

    echo "Container ID: $container_id"

    # docker exec bypasses the image ENTRYPOINT, so we must activate the
    # venv explicitly (matching the ENTRYPOINT in the Dockerfile).
    local activate="source /workspace/.venv/bin/activate"

    if [[ $# -eq 0 ]]; then
        echo "Attaching interactive shell..."
        set -x
        docker exec -it "$container_id" /bin/bash -c "$activate && exec bash"
        set +x
    else
        echo "Executing command: $*"
        local escaped_cmd
        escaped_cmd=$(printf '%q ' "$@")
        set -x
        docker exec -it "$container_id" /bin/bash -c "$activate && $escaped_cmd"
        set +x
    fi
}

cmd_stop() {
    require_vars CONTAINER_NAME

    echo ""
    echo "=== Stop ==="
    echo "Container: $CONTAINER_NAME"

    local container_id
    container_id=$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}' | head -1)

    if [[ -z "$container_id" ]]; then
        echo "No running container found with name '$CONTAINER_NAME'"
        exit 0
    fi

    echo "Stopping container $container_id..."
    docker stop "$container_id"
    echo "Done: stopped $CONTAINER_NAME"
}

cmd_flatten() {
    # -------------------------------------------------------------------------
    # Flatten a Docker image into a single layer to speed up container startup.
    #
    # Many-layered images (100+) cause slow overlay2 mount setup on each
    # `docker run`. Flattening reduces this to 1 layer.
    #
    # Usage:
    #   docker-ops.sh flatten                 # Flatten DOCKER_IMAGE in-place
    #   docker-ops.sh flatten -o flat:tag     # Flatten into a new image name
    # -------------------------------------------------------------------------
    require_vars DOCKER_IMAGE

    local output_image="${DOCKER_IMAGE}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -o) output_image="$2"; shift 2 ;;
            *) echo "Unknown option: $1"; echo "Usage: docker-ops.sh flatten [-o IMAGE:TAG]"; exit 1 ;;
        esac
    done

    local layer_count
    layer_count=$(docker history -q "$DOCKER_IMAGE" 2>/dev/null | wc -l)

    echo ""
    echo "=== Flatten ==="
    echo "Source:     $DOCKER_IMAGE ($layer_count layers)"
    echo "Output:     $output_image"
    echo ""

    # Export the image filesystem and re-import as a single layer.
    # Preserve WORKDIR, ENTRYPOINT, CMD from the original image.
    # NOTE: ENV variables are lost — they must be set at runtime via env.sh.
    local tmp_container="flatten-tmp-$$"

    echo "Creating temporary container..."
    docker create --name "$tmp_container" "$DOCKER_IMAGE" /bin/true >/dev/null

    echo "Exporting and re-importing as single layer..."
    # Grab original config for the commit message
    local original_entrypoint
    original_entrypoint=$(docker inspect --format='{{json .Config.Entrypoint}}' "$DOCKER_IMAGE" 2>/dev/null)
    local original_cmd
    original_cmd=$(docker inspect --format='{{json .Config.Cmd}}' "$DOCKER_IMAGE" 2>/dev/null)
    local original_workdir
    original_workdir=$(docker inspect --format='{{.Config.WorkingDir}}' "$DOCKER_IMAGE" 2>/dev/null)

    # Build --change flags to preserve config
    local -a change_flags=()
    [[ -n "$original_workdir" ]] && change_flags+=(--change "WORKDIR $original_workdir")
    [[ "$original_entrypoint" != "null" && -n "$original_entrypoint" ]] && change_flags+=(--change "ENTRYPOINT $original_entrypoint")
    [[ "$original_cmd" != "null" && -n "$original_cmd" ]] && change_flags+=(--change "CMD $original_cmd")

    docker export "$tmp_container" | docker import "${change_flags[@]}" - "$output_image"
    docker rm "$tmp_container" >/dev/null

    local new_layer_count
    new_layer_count=$(docker history -q "$output_image" 2>/dev/null | wc -l)

    echo ""
    echo "Done: $output_image ($new_layer_count layer, was $layer_count)"
    echo ""
    echo "Note: ENV variables from the original image are lost in flatten."
    echo "They are set at runtime via env.sh (source env.sh)."
}

# ==============================================================================
# Usage
# ==============================================================================

usage() {
    cat <<EOF
Docker Operations Script

Usage: docker-ops.sh [command] [options]

Commands:
  pull                          Pull container from registry
  build [--clean] [--base IMG]  Build container locally
  push [TAG]                    Push to registry (optional alternate tag)
  squash [-o FILE]              Create squashfs for Slurm (requires enroot)
  flatten [-o IMAGE:TAG]        Flatten image to 1 layer (faster startup)
  run [options] [CMD]           Run container (default: interactive bash)
  attach [CMD]                  Attach to running container
  stop                          Stop running container

Run options:
  --no-code-mount    Do not mount local code into container
  --no-ssh           Do not mount ~/.ssh
  --no-claude        Do not mount claude binary

Configuration:
  Set PROTEINA_ENV to use a custom env file (default: .env).
  If .env is missing, .env_example is copied as a starting point.

  Key variables (see .env_example for full list):
    DOCKER_IMAGE       Full image name with registry and tag
    REGISTRY           Container registry URL
    REGISTRY_USER      Registry username
    GITLAB_TOKEN       Registry auth token
    LOCAL_DATA_PATH    Host data path
    DOCKER_DATA_PATH   Container data mount path
    DOCKER_REPO_PATH   Container code mount path
    CONTAINER_NAME     Name for the running container
    DOCKERFILE_PATH    Path to Dockerfile (for build)

Examples:
  docker-ops.sh pull
  docker-ops.sh build
  docker-ops.sh build --clean
  docker-ops.sh push
  docker-ops.sh push v1.0
  docker-ops.sh run
  docker-ops.sh run python train.py
  docker-ops.sh run --no-code-mount bash
  docker-ops.sh attach
  docker-ops.sh attach ls -la
  docker-ops.sh stop
  docker-ops.sh squash
EOF
}

# ==============================================================================
# Main
# ==============================================================================

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

load_env

case "$1" in
    pull)    shift; cmd_pull "$@" ;;
    build)   shift; cmd_build "$@" ;;
    push)    shift; cmd_push "$@" ;;
    squash)  shift; cmd_squash "$@" ;;
    flatten) shift; cmd_flatten "$@" ;;
    run)     shift; cmd_run "$@" ;;
    attach)  shift; cmd_attach "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    *)       usage ;;
esac
