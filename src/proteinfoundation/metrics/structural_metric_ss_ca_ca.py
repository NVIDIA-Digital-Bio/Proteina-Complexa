import biotite.structure as struc
import biotite.structure.io as strucio
import torch

from proteinfoundation.utils.pdb_utils import from_pdb_file


def load_alpha_carbon_coordinates(pdb_file):
    prot = from_pdb_file(pdb_file)
    mask = torch.tensor(prot.atom_mask, dtype=torch.bool)  # [n, 37]
    coors_atom37 = torch.tensor(prot.atom_positions, dtype=torch.float32)  # [n, 37, 3]
    mask_ca = mask[:, 1]  # [n]
    return coors_atom37[mask_ca, 1, :]  # [n_unmasked, 3]


def compute_ca_metrics(pdb_path):
    try:
        coors = load_alpha_carbon_coordinates(pdb_path)  # [n, 3]
        consecutive_ca_ca_distances = torch.norm(coors[1:, :] - coors[:-1, :], dim=-1)  # [n-1]
        pairwise_ca_ca_distances = torch.norm(coors[None, :, :] - coors[:, None, :], dim=-1)  # [n, n]
        num_collisions = torch.sum((pairwise_ca_ca_distances > 0.01) & (pairwise_ca_ca_distances < 2.0)) / 2.0
        # The greater than is to avoid diagonal elements which do not count as collisions
        return {
            "ca_ca_dist_avg": torch.mean(consecutive_ca_ca_distances),
            "ca_ca_dist_median": torch.median(consecutive_ca_ca_distances),
            "ca_ca_dist_std": torch.std(consecutive_ca_ca_distances),
            "ca_ca_dist_min": torch.min(consecutive_ca_ca_distances),
            "ca_ca_dist_max": torch.max(consecutive_ca_ca_distances),
            "ca_ca_collisions(2A)": num_collisions,
        }
    except Exception as e:
        print(f"Error in ca-ca metrics {e}")
        return {
            "ca_ca_dist_avg": 0.0,
            "ca_ca_dist_median": 0.0,
            "ca_ca_dist_std": 0.0,
            "ca_ca_dist_min": 0.0,
            "ca_ca_dist_max": 0.0,
            "ca_ca_collisions(2A)": 0.0,
        }


def compute_ss_metrics(pdb_path):
    # Leaving in in case at some point we want to go back to mdtraj
    # try:
    #     traj = md.load(pdb_path)
    #     pdb_ss = md.compute_dssp(traj, simplified=True)
    #     pdb_coil_percent = np.mean(pdb_ss == "C")
    #     pdb_helix_percent = np.mean(pdb_ss == "H")
    #     pdb_strand_percent = np.mean(pdb_ss == "E")
    #     pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
    #     pdb_rg = md.compute_rg(traj)[0]
    # except Exception as e:
    #     pdb_ss_percent = 0.0
    #     pdb_coil_percent = 0.0
    #     pdb_helix_percent = 0.0
    #     pdb_strand_percent = 0.0

    try:
        stack = strucio.load_structure(pdb_path)
        sse = struc.annotate_sse(stack)
        a = (sse == "a").sum()  # num alpha
        b = (sse == "b").sum()  # num beta
        c = (sse == "c").sum()  # num coil
        tot = a + b + c
    except Exception:
        a = 0.0
        b = 0.0
        c = 0.0
        tot = 1.0

    return {
        "biot_alpha": a / tot,
        "biot_beta": b / tot,
        "biot_coil": c / tot,
    }


def compute_structural_metrics(pdb_path):
    """Computes a bunch of validation metrics, returns them as a dictionary."""
    metrics_ss = compute_ss_metrics(pdb_path)
    metrics_ca_ca = compute_ca_metrics(pdb_path)
    return {**metrics_ss, **metrics_ca_ca}
