#!/bin/bash
# =============================================================================
# Protein Foundation Models - Startup Script
# =============================================================================
# This script downloads all required model weights for the protein foundation
# models project. Run this from the project root directory.
# =============================================================================

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║          🧬  Protein Foundation Models - Setup Wizard  🧬         ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${WHITE}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_progress() {
    echo -e "${CYAN}→${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed. Please install it and try again."
        return 1
    fi
    return 0
}

# =============================================================================
# Download Functions
# =============================================================================

download_pmpnn_weights() {
    print_section "Downloading ProteinMPNN Weights"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "community_models/ProteinMPNN" ]; then
        print_error "ProteinMPNN directory not found at $PROJECT_ROOT/community_models/ProteinMPNN"
        return 1
    fi
    
    cd community_models/ProteinMPNN
    mkdir -p ca_model_weights vanilla_model_weights
    
    local downloaded=0
    local skipped=0
    
    # Download CA model weights
    print_progress "Checking CA model weights..."
    cd ca_model_weights
    local ca_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt")
    for file in "${ca_files[@]}"; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            print_info "  ✓ $file already exists, skipping"
            skipped=$((skipped + 1))
        else
            print_info "  Downloading $file..."
            wget -q --show-progress "https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/$file" -O "$file"
            downloaded=$((downloaded + 1))
        fi
    done
    cd ..
    
    # Download vanilla model weights
    print_progress "Checking vanilla model weights..."
    cd vanilla_model_weights
    local vanilla_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")
    for file in "${vanilla_files[@]}"; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            print_info "  ✓ $file already exists, skipping"
            skipped=$((skipped + 1))
        else
            print_info "  Downloading $file..."
            wget -q --show-progress "https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/$file" -O "$file"
            downloaded=$((downloaded + 1))
        fi
    done
    cd ..
    
    if [ $downloaded -eq 0 ]; then
        print_success "ProteinMPNN weights already installed ($skipped files)"
    else
        print_success "ProteinMPNN weights: downloaded $downloaded, skipped $skipped"
    fi
    cd "$PROJECT_ROOT"
}

download_ligmpnn_weights() {
    print_section "Downloading LigandMPNN Weights"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "community_models/LigandMPNN" ]; then
        print_error "LigandMPNN directory not found at $PROJECT_ROOT/community_models/LigandMPNN"
        return 1
    fi
    
    cd community_models/LigandMPNN
    mkdir -p model_params
    
    # Check if weights already exist
    local required_files=(
        "proteinmpnn_v_48_002.pt"
        "proteinmpnn_v_48_020.pt"
        "ligandmpnn_v_32_010_25.pt"
        "ligandmpnn_v_32_020_25.pt"
        "solublempnn_v_48_020.pt"
    )
    
    local all_exist=true
    for file in "${required_files[@]}"; do
        if [ ! -f "model_params/$file" ] || [ ! -s "model_params/$file" ]; then
            all_exist=false
            break
        fi
    done
    
    if [ "$all_exist" = true ]; then
        print_success "LigandMPNN weights already installed, skipping download"
        cd "$PROJECT_ROOT"
        return 0
    fi
    
    if [ -f "get_model_params.sh" ]; then
        print_progress "Running LigandMPNN weight download script..."
        bash get_model_params.sh "./model_params"
        print_success "LigandMPNN weights downloaded successfully!"
    else
        print_error "get_model_params.sh not found in LigandMPNN directory"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
}

download_af2_weights() {
    print_section "Downloading AlphaFold2 Weights"
    
    cd "$PROJECT_ROOT"
    
    local params_dir="./community_models/ckpts/AF2"
    local params_file="${params_dir}/alphafold_params_2022-12-06.tar"
    
    # Check if weights already exist
    if [ -f "${params_dir}/params_model_5_ptm.npz" ] && [ -s "${params_dir}/params_model_5_ptm.npz" ]; then
        print_success "AlphaFold2 weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating params directory at community_models/ckpts/AF2/..."
    mkdir -p "${params_dir}" || { print_error "Failed to create weights directory"; return 1; }
    
    print_progress "Downloading AlphaFold2 weights (this may take a while, ~5GB)..."
    wget --show-progress -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { 
        print_error "Failed to download AlphaFold2 weights"
        return 1
    }
    
    if [ ! -s "${params_file}" ]; then
        print_error "Could not locate downloaded AlphaFold2 weights"
        return 1
    fi
    
    print_progress "Verifying archive integrity..."
    tar tf "${params_file}" >/dev/null 2>&1 || { 
        print_error "Corrupt AlphaFold2 weights download"
        return 1
    }
    
    print_progress "Extracting AlphaFold2 weights..."
    tar -xvf "${params_file}" -C "${params_dir}" || { 
        print_error "Failed to extract AlphaFold2 weights"
        return 1
    }
    
    if [ ! -f "${params_dir}/params_model_5_ptm.npz" ]; then
        print_error "Could not locate extracted AlphaFold2 weights"
        return 1
    fi
    
    print_progress "Cleaning up archive..."
    rm "${params_file}" || print_warning "Failed to remove AlphaFold2 weights archive"
    
    print_success "AlphaFold2 weights downloaded to community_models/ckpts/AF2/ successfully!"
    cd "$PROJECT_ROOT"
}

download_complexa_weights() {
    print_section "Downloading Complexa Weights"
    
    cd "$PROJECT_ROOT"
    
    local ckpt_dir="./ckpts"
    local fm_ckpt="${ckpt_dir}/complexa.ckpt"
    local ae_ckpt="${ckpt_dir}/complexa_ae.ckpt"
    
    # Check if weights already exist
    if [ -f "$fm_ckpt" ] && [ -s "$fm_ckpt" ] && [ -f "$ae_ckpt" ] && [ -s "$ae_ckpt" ]; then
        print_success "Complexa weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating ckpts directory..."
    mkdir -p "${ckpt_dir}" || { print_error "Failed to create ckpts directory"; return 1; }
    
    local ngc_model="proteina_complexa"
    local ngc_base="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/${ngc_model}/1.0/files?redirect=true&path="

    if [ ! -f "$fm_ckpt" ] || [ ! -s "$fm_ckpt" ]; then
        print_progress "Downloading complexa.ckpt (flow matching model)..."
        wget --content-disposition --show-progress -O "$fm_ckpt" \
            "${ngc_base}complexa.ckpt" || {
            print_error "Failed to download complexa.ckpt"
            return 1
        }
        [ -s "$fm_ckpt" ] || { print_error "Downloaded complexa.ckpt is empty"; return 1; }
    fi

    if [ ! -f "$ae_ckpt" ] || [ ! -s "$ae_ckpt" ]; then
        print_progress "Downloading complexa_ae.ckpt (autoencoder)..."
        wget --content-disposition --show-progress -O "$ae_ckpt" \
            "${ngc_base}complexa_ae.ckpt" || {
            print_error "Failed to download complexa_ae.ckpt"
            return 1
        }
        [ -s "$ae_ckpt" ] || { print_error "Downloaded complexa_ae.ckpt is empty"; return 1; }
    fi

    print_success "Complexa weights downloaded to ckpts/ successfully!"
}

download_complexa_ligand_weights() {
    print_section "Downloading Complexa Ligand Weights"
    
    cd "$PROJECT_ROOT"
    
    local ckpt_dir="./ckpts"
    local fm_ckpt="${ckpt_dir}/complexa_ligand.ckpt"
    local ae_ckpt="${ckpt_dir}/complexa_ligand_ae.ckpt"
    
    # Check if weights already exist
    if [ -f "$fm_ckpt" ] && [ -s "$fm_ckpt" ] && [ -f "$ae_ckpt" ] && [ -s "$ae_ckpt" ]; then
        print_success "Complexa Ligand weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating ckpts directory..."
    mkdir -p "${ckpt_dir}" || { print_error "Failed to create ckpts directory"; return 1; }
    
    local ngc_model="proteina_complexa_ligand"
    local ngc_base="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/${ngc_model}/1.0/files?redirect=true&path="

    if [ ! -f "$fm_ckpt" ] || [ ! -s "$fm_ckpt" ]; then
        print_progress "Downloading complexa_ligand.ckpt (flow matching model)..."
        wget --content-disposition --show-progress -O "$fm_ckpt" \
            "${ngc_base}complexa_ligand.ckpt" || {
            print_error "Failed to download complexa_ligand.ckpt"
            return 1
        }
        [ -s "$fm_ckpt" ] || { print_error "Downloaded complexa_ligand.ckpt is empty"; return 1; }
    fi

    if [ ! -f "$ae_ckpt" ] || [ ! -s "$ae_ckpt" ]; then
        print_progress "Downloading complexa_ligand_ae.ckpt (autoencoder)..."
        wget --content-disposition --show-progress -O "$ae_ckpt" \
            "${ngc_base}complexa_ligand_ae.ckpt" || {
            print_error "Failed to download complexa_ligand_ae.ckpt"
            return 1
        }
        [ -s "$ae_ckpt" ] || { print_error "Downloaded complexa_ligand_ae.ckpt is empty"; return 1; }
    fi

    print_success "Complexa Ligand weights downloaded to ckpts/ successfully!"
}

download_complexa_ame_weights() {
    print_section "Downloading Complexa AME (Enzyme) Weights"
    
    cd "$PROJECT_ROOT"
    
    local ckpt_dir="./ckpts"
    local fm_ckpt="${ckpt_dir}/complexa_ame.ckpt"
    local ae_ckpt="${ckpt_dir}/complexa_ame_ae.ckpt"
    
    # Check if weights already exist
    if [ -f "$fm_ckpt" ] && [ -s "$fm_ckpt" ] && [ -f "$ae_ckpt" ] && [ -s "$ae_ckpt" ]; then
        print_success "Complexa AME weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating ckpts directory..."
    mkdir -p "${ckpt_dir}" || { print_error "Failed to create ckpts directory"; return 1; }
    
    local ngc_model="proteina_complexa_ame"
    local ngc_base="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/${ngc_model}/1.0/files?redirect=true&path="

    if [ ! -f "$fm_ckpt" ] || [ ! -s "$fm_ckpt" ]; then
        print_progress "Downloading complexa_ame.ckpt (flow matching model)..."
        wget --content-disposition --show-progress -O "$fm_ckpt" \
            "${ngc_base}complexa_ame.ckpt" || {
            print_error "Failed to download complexa_ame.ckpt"
            return 1
        }
        [ -s "$fm_ckpt" ] || { print_error "Downloaded complexa_ame.ckpt is empty"; return 1; }
    fi

    if [ ! -f "$ae_ckpt" ] || [ ! -s "$ae_ckpt" ]; then
        print_progress "Downloading complexa_ame_ae.ckpt (autoencoder)..."
        wget --content-disposition --show-progress -O "$ae_ckpt" \
            "${ngc_base}complexa_ame_ae.ckpt" || {
            print_error "Failed to download complexa_ame_ae.ckpt"
            return 1
        }
        [ -s "$ae_ckpt" ] || { print_error "Downloaded complexa_ame_ae.ckpt is empty"; return 1; }
    fi

    print_success "Complexa AME weights downloaded to ckpts/ successfully!"
}

download_esm2_weights() {
    print_section "Downloading ESM2 Weights"
    
    cd "$PROJECT_ROOT"
    
    local params_dir="./community_models/ckpts/ESM2"
    
    # Check if weights already exist by looking for model files
    if [ -d "${params_dir}/huggingface/hub" ] && [ "$(find "${params_dir}/huggingface/hub" -name "*.safetensors" 2>/dev/null | head -1)" ]; then
        print_success "ESM2 weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating ESM2 params directory at community_models/ckpts/ESM2/..."
    mkdir -p "${params_dir}" || { print_error "Failed to create weights directory"; return 1; }
    
    # Check for Python and transformers
    if ! command -v python &> /dev/null; then
        print_error "Python is not available. Please activate your environment first."
        return 1
    fi
    
    # Check for HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        print_warning "HF_TOKEN not set. If download fails with HTTP 429, set your token:"
        print_info "  export HF_TOKEN='your_huggingface_token'"
        print_info "  Get a free token at: https://huggingface.co/settings/tokens"
    else
        print_info "Using HF_TOKEN for authentication"
    fi
    
    print_progress "Downloading ESM2 model (facebook/esm2_t33_650M_UR50D)..."
    print_info "This downloads ~2.6GB and requires internet access."
    
    # Use the Python download script
    python "$PROJECT_ROOT/script_utils/download/download_esm_model.py" \
        --cache-dir "${params_dir}" \
        ${HF_TOKEN:+--token "$HF_TOKEN"} || {
        print_error "Failed to download ESM2 weights"
        print_info "Try running on a node with internet access, or set HF_TOKEN if rate limited."
        return 1
    }
    
    print_success "ESM2 weights downloaded to community_models/ckpts/ESM2/ successfully!"
    cd "$PROJECT_ROOT"
}

download_rf3_weights() {
    print_section "Downloading RoseTTAFold3 Weights"
    
    cd "$PROJECT_ROOT"
    
    local params_dir="./community_models/ckpts/RF3"
    local ckpt_file="${params_dir}/rf3_foundry_01_24_latest_remapped.ckpt"
    
    # Check if weights already exist
    if [ -f "$ckpt_file" ] && [ -s "$ckpt_file" ]; then
        print_success "RF3 weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating RF3 params directory at community_models/ckpts/RF3/..."
    mkdir -p "${params_dir}" || { print_error "Failed to create weights directory"; return 1; }
    
    # Download latest RF3 checkpoint (best performance, trained with data until 1/2024)
    print_progress "Downloading RF3 latest checkpoint (rf3_foundry_01_24_latest_remapped.ckpt)..."
    wget --show-progress -O "$ckpt_file" \
        "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt" || { 
        print_error "Failed to download RF3 checkpoint"
        return 1
    }
    [ -s "$ckpt_file" ] || { 
        print_error "Could not locate downloaded RF3 checkpoint"
        return 1
    }
    
    # -------------------------------------------------------------------------
    # Alternative RF3 checkpoints (uncomment to download):
    # -------------------------------------------------------------------------
    # RF3 preprint checkpoint trained with data until 9/2021
    # print_progress "Downloading RF3 preprint 9/21 checkpoint..."
    # wget --show-progress -O "${params_dir}/rf3_foundry_09_21_preprint_remapped.ckpt" \
    #     "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_09_21_preprint_remapped.ckpt"
    
    # RF3 preprint checkpoint trained with data until 1/2024
    # print_progress "Downloading RF3 preprint 1/24 checkpoint..."
    # wget --show-progress -O "${params_dir}/rf3_foundry_01_24_preprint_remapped.ckpt" \
    #     "https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_preprint_remapped.ckpt"
    # -------------------------------------------------------------------------
    
    print_success "RoseTTAFold3 weights downloaded to community_models/ckpts/RF3/ successfully!"
    cd "$PROJECT_ROOT"
}

# =============================================================================
# Status Check Functions
# =============================================================================

check_pmpnn_status() {
    local missing=()
    local ca_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt")
    local vanilla_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")
    
    for file in "${ca_files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/ProteinMPNN/ca_model_weights/$file" ] && missing+=("ca_model_weights/$file")
    done
    for file in "${vanilla_files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/ProteinMPNN/vanilla_model_weights/$file" ] && missing+=("vanilla_model_weights/$file")
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (community_models/ProteinMPNN/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (community_models/ProteinMPNN/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_ligmpnn_status() {
    local missing=()
    local files=(
        "proteinmpnn_v_48_002.pt"
        "proteinmpnn_v_48_010.pt"
        "proteinmpnn_v_48_020.pt"
        "proteinmpnn_v_48_030.pt"
        "ligandmpnn_v_32_005_25.pt"
        "ligandmpnn_v_32_010_25.pt"
        "ligandmpnn_v_32_020_25.pt"
        "ligandmpnn_v_32_030_25.pt"
        "per_residue_label_membrane_mpnn_v_48_020.pt"
        "global_label_membrane_mpnn_v_48_020.pt"
        "solublempnn_v_48_002.pt"
        "solublempnn_v_48_010.pt"
        "solublempnn_v_48_020.pt"
        "solublempnn_v_48_030.pt"
        "ligandmpnn_sc_v_32_002_16.pt"
    )
    
    for file in "${files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/LigandMPNN/model_params/$file" ] && missing+=("$file")
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (community_models/LigandMPNN/model_params/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (community_models/LigandMPNN/model_params/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_af2_status() {
    local missing=()
    local af2_params_dir="$PROJECT_ROOT/community_models/ckpts/AF2"
    local files=(
        "params_model_1.npz"
        "params_model_1_ptm.npz"
        "params_model_1_multimer_v3.npz"
        "params_model_2.npz"
        "params_model_2_ptm.npz"
        "params_model_2_multimer_v3.npz"
        "params_model_3.npz"
        "params_model_3_ptm.npz"
        "params_model_3_multimer_v3.npz"
        "params_model_4.npz"
        "params_model_4_ptm.npz"
        "params_model_4_multimer_v3.npz"
        "params_model_5.npz"
        "params_model_5_ptm.npz"
        "params_model_5_multimer_v3.npz"
    )
    
    for file in "${files[@]}"; do
        [ ! -f "$af2_params_dir/$file" ] && missing+=("$file")
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (community_models/ckpts/AF2/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (community_models/ckpts/AF2/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_rf3_status() {
    local missing=()
    local files=(
        "rf3_foundry_01_24_latest_remapped.ckpt"
    )
    
    for file in "${files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/ckpts/RF3/$file" ] && missing+=("$file")
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (community_models/ckpts/RF3/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (community_models/ckpts/RF3/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_esm2_status() {
    local esm2_dir="$PROJECT_ROOT/community_models/ckpts/ESM2"
    
    # Check if directory exists and contains the ESM2 model
    if [ -d "${esm2_dir}" ]; then
        # Check for the models--facebook--esm2_t33_650M_UR50D directory (HF cache structure)
        if [ -d "${esm2_dir}/models--facebook--esm2_t33_650M_UR50D" ]; then
            # Look for the large blob file (model weights ~2.5GB)
            local blob_files=$(find "${esm2_dir}/models--facebook--esm2_t33_650M_UR50D" -type f -size +100M 2>/dev/null | head -1)
            if [ -n "$blob_files" ]; then
                echo -e "${GREEN}✓ Installed (community_models/ckpts/ESM2/)${NC}"
                return
            fi
        fi
        # Fallback: look for any large files (>100MB) indicating model weights
        local large_files=$(find "${esm2_dir}" -type f -size +100M 2>/dev/null | head -1)
        if [ -n "$large_files" ]; then
            echo -e "${GREEN}✓ Installed (community_models/ckpts/ESM2/)${NC}"
            return
        fi
    fi
    echo -e "${YELLOW}○ Not installed (community_models/ckpts/ESM2/)${NC}"
}

check_complexa_status() {
    local missing=()
    local ckpt_dir="$PROJECT_ROOT/ckpts"
    
    [ ! -f "$ckpt_dir/complexa.ckpt" ] && missing+=("complexa.ckpt (flow matching model)")
    [ ! -f "$ckpt_dir/complexa_ae.ckpt" ] && missing+=("complexa_ae.ckpt (autoencoder)")
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (ckpts/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (ckpts/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_complexa_ligand_status() {
    local missing=()
    local ckpt_dir="$PROJECT_ROOT/ckpts"
    
    [ ! -f "$ckpt_dir/complexa_ligand.ckpt" ] && missing+=("complexa_ligand.ckpt (flow matching model)")
    [ ! -f "$ckpt_dir/complexa_ligand_ae.ckpt" ] && missing+=("complexa_ligand_ae.ckpt (autoencoder)")
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (ckpts/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (ckpts/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_complexa_ame_status() {
    local missing=()
    local ckpt_dir="$PROJECT_ROOT/ckpts"
    
    [ ! -f "$ckpt_dir/complexa_ame.ckpt" ] && missing+=("complexa_ame.ckpt (flow matching model)")
    [ ! -f "$ckpt_dir/complexa_ame_ae.ckpt" ] && missing+=("complexa_ame_ae.ckpt (autoencoder)")
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (ckpts/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (ckpts/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

print_status() {
    print_section "Current Installation Status"
    echo -e "  ${BOLD}Complexa Models (Required):${NC}"
    echo -e "    Complexa (Protein): $(check_complexa_status)"
    echo -e "    Complexa (Ligand):  $(check_complexa_ligand_status)"
    echo -e "    Complexa (AME):     $(check_complexa_ame_status)"
    echo -e ""
    echo -e "  ${BOLD}Core Models:${NC}"
    echo -e "    ProteinMPNN:     $(check_pmpnn_status)"
    echo -e "    LigandMPNN:      $(check_ligmpnn_status)"
    echo -e "    AlphaFold2:      $(check_af2_status)"
    echo -e "    ESM2:            $(check_esm2_status)"
    echo -e "    RF3:             $(check_rf3_status)"
    echo ""
}

# =============================================================================
# Interactive Menu
# =============================================================================

show_menu() {
    echo -e "${BOLD}${WHITE}What would you like to download?${NC}\n"
    echo -e "  ${BOLD}Complexa Models:${NC}"
    echo -e "  ${CYAN}p${NC}) Download ${BOLD}Complexa Protein${NC} weights (protein binder design)"
    echo -e "  ${CYAN}l${NC}) Download ${BOLD}Complexa Ligand${NC} weights (ligand binder design)"
    echo -e "  ${CYAN}e${NC}) Download ${BOLD}Complexa AME${NC} weights (enzyme/motif scaffolding)"
    echo -e "  ${CYAN}a${NC}) Download ${BOLD}all Complexa${NC} weights (protein + ligand + AME)"
    echo -e ""
    echo -e "  ${BOLD}Community Models:${NC}"
    echo -e "  ${CYAN}1${NC}) Download ${BOLD}all community${NC} weights (ProteinMPNN + LigandMPNN + AF2 + ESM2 + RF3)"
    echo -e "  ${CYAN}2${NC}) Download ${BOLD}ProteinMPNN${NC} weights only (~50MB)"
    echo -e "  ${CYAN}3${NC}) Download ${BOLD}LigandMPNN${NC} weights only (~500MB)"
    echo -e "  ${CYAN}4${NC}) Download ${BOLD}AlphaFold2${NC} weights only (~5GB)"
    echo -e "  ${CYAN}5${NC}) Download ${BOLD}ESM2${NC} weights only (~2.6GB) ${YELLOW}[requires HF_TOKEN if rate limited]${NC}"
    echo -e "  ${CYAN}6${NC}) Download ${BOLD}RoseTTAFold3${NC} weights (~2.5GB)"
    echo -e "  ${CYAN}7${NC}) Download ${BOLD}everything${NC} (all Complexa + community)"
    echo -e ""
    echo -e "  ${BOLD}Other:${NC}"
    echo -e "  ${CYAN}s${NC}) Check installation ${BOLD}status${NC}"
    echo -e "  ${CYAN}q${NC}) ${BOLD}Quit${NC}"
    echo ""
    echo -ne "${YELLOW}Enter your choice [p, l, e, a, 1-7, s, q]: ${NC}"
}

run_interactive() {
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            p|P)
                download_complexa_weights || true
                ;;
            l|L)
                download_complexa_ligand_weights || true
                ;;
            e|E)
                download_complexa_ame_weights || true
                ;;
            a|A)
                download_complexa_weights || true
                download_complexa_ligand_weights || true
                download_complexa_ame_weights || true
                print_section "Complexa Downloads Complete! 🎉"
                print_status
                ;;
            1)
                download_pmpnn_weights
                download_ligmpnn_weights
                download_af2_weights
                download_esm2_weights
                download_rf3_weights
                print_section "Community Downloads Complete! 🎉"
                print_status
                ;;
            2)
                download_pmpnn_weights
                ;;
            3)
                download_ligmpnn_weights
                ;;
            4)
                download_af2_weights
                ;;
            5)
                download_esm2_weights
                ;;
            6)
                download_rf3_weights
                ;;
            7)
                download_complexa_weights || true
                download_complexa_ligand_weights || true
                download_complexa_ame_weights || true
                download_pmpnn_weights
                download_ligmpnn_weights
                download_af2_weights
                download_esm2_weights
                download_rf3_weights
                print_section "All Downloads Complete! 🎉"
                print_status
                ;;
            s|S)
                print_status
                ;;
            q|Q)
                echo -e "\n${GREEN}Enjoy designing! 👋${NC}\n"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        echo -ne "${YELLOW}Press Enter to continue...${NC}"
        read -r
        clear
        print_banner
    done
}

# =============================================================================
# CLI Mode
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Complexa Models:"
    echo "  --complexa        Download Complexa Protein weights (protein binder design)"
    echo "  --complexa-ligand Download Complexa Ligand weights (ligand binder design)"
    echo "  --complexa-ame    Download Complexa AME weights (enzyme/motif scaffolding)"
    echo "  --complexa-all    Download all Complexa weights (protein + ligand + AME)"
    echo ""
    echo "Community Model Options:"
    echo "  --all           Download all community model weights (ProteinMPNN, LigandMPNN, AF2, ESM2, RF3)"
    echo "  --pmpnn         Download ProteinMPNN weights"
    echo "  --ligmpnn       Download LigandMPNN weights"
    echo "  --af2           Download AlphaFold2 weights"
    echo "  --esm2          Download ESM2 650M weights (requires HF_TOKEN if rate limited)"
    echo "  --rf3           Download RoseTTAFold3 weights"
    echo ""
    echo "Other Model Options:"
    echo "  --everything    Download all models (Complexa + community)"
    echo ""
    echo "Other Options:"
    echo "  --status        Show installation status"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_TOKEN        Hugging Face token for ESM2 650M download (avoids rate limits)"
    echo ""
    echo "If no options are provided, runs in interactive mode."
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Check for required commands
    check_command wget || exit 1
    check_command tar || exit 1
    
    # Change to project root
    cd "$PROJECT_ROOT"
    print_info "Working directory: $PROJECT_ROOT"
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        clear
        print_banner
        print_status
        run_interactive
    else
        # CLI mode
        print_banner
        
        for arg in "$@"; do
            case $arg in
                --complexa)
                    download_complexa_weights || true
                    ;;
                --complexa-ligand)
                    download_complexa_ligand_weights || true
                    ;;
                --complexa-ame)
                    download_complexa_ame_weights || true
                    ;;
                --complexa-all)
                    download_complexa_weights || true
                    download_complexa_ligand_weights || true
                    download_complexa_ame_weights || true
                    print_section "Complexa Downloads Complete! 🎉"
                    ;;
                --all)
                    download_pmpnn_weights
                    download_ligmpnn_weights
                    download_af2_weights
                    download_esm2_weights
                    download_rf3_weights
                    print_section "Community Downloads Complete! 🎉"
                    ;;
                --everything)
                    download_complexa_weights || true
                    download_complexa_ligand_weights || true
                    download_complexa_ame_weights || true
                    download_pmpnn_weights
                    download_ligmpnn_weights
                    download_af2_weights
                    download_esm2_weights
                    download_rf3_weights
                    print_section "All Downloads Complete! 🎉"
                    ;;
                --pmpnn)
                    download_pmpnn_weights
                    ;;
                --ligmpnn)
                    download_ligmpnn_weights
                    ;;
                --af2)
                    download_af2_weights
                    ;;
                --esm2)
                    download_esm2_weights
                    ;;
                --rf3)
                    download_rf3_weights
                    ;;
                --status)
                    print_status
                    ;;
                --help|-h)
                    show_help
                    exit 0
                    ;;
                *)
                    print_error "Unknown option: $arg"
                    show_help
                    exit 1
                    ;;
            esac
        done
        
        print_status
    fi
}

main "$@"

