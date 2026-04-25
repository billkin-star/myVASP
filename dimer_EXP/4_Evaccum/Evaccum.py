import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_atom_list(atom_str):
    """
    Parse atom index string like:
      "1,2,3,4"
      "1-4"
      "1-4,7,9-12"
    Returns sorted unique 1-based atom indices.
    """
    atoms = set()
    for part in atom_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            atoms.update(range(a, b + 1))
        else:
            atoms.add(int(part))
    return sorted(atoms)


def read_poscar_lattice(poscar="POSCAR"):
    with open(poscar, "r") as f:
        lines = f.readlines()
    scale = float(lines[1].split()[0])
    lattice = []
    for i in range(2, 5):
        lattice.append([float(x) for x in lines[i].split()[:3]])
    lattice = np.array(lattice) * scale
    return lattice


def read_locpot(locpot="LOCPOT"):
    """
    Minimal LOCPOT parser for scalar potential.
    Assumes ISPIN=1 scalar potential block.
    Returns:
        lattice (3,3)
        species_counts
        grid (nx, ny, nz)
        pot (nx, ny, nz)
    """
    with open(locpot, "r") as f:
        lines = f.readlines()

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale

    tokens5 = lines[5].split()
    if all(t.replace("-", "").isdigit() for t in tokens5):
        counts = [int(x) for x in lines[5].split()]
        coord_start = 7
    else:
        counts = [int(x) for x in lines[6].split()]
        coord_start = 8

    natoms = sum(counts)

    line = lines[coord_start - 1].strip().lower()
    if line.startswith("s"):
        coord_start += 1

    grid_line_idx = coord_start + natoms
    while grid_line_idx < len(lines) and len(lines[grid_line_idx].split()) == 0:
        grid_line_idx += 1

    nx, ny, nz = [int(x) for x in lines[grid_line_idx].split()]
    ngrid = nx * ny * nz

    vals = []
    idx = grid_line_idx + 1
    while idx < len(lines) and len(vals) < ngrid:
        parts = lines[idx].split()
        vals.extend([float(x) for x in parts])
        idx += 1

    vals = np.array(vals[:ngrid], dtype=float)
    pot = vals.reshape((nx, ny, nz), order="F")

    return lattice, counts, (nx, ny, nz), pot


def planar_average_z(pot, lattice):
    """
    Average over x-y, return z coordinates in Angstrom and V(z) in eV.
    """
    nx, ny, nz = pot.shape
    vz = pot.mean(axis=(0, 1))
    cvec = lattice[2]
    c = np.linalg.norm(cvec)
    z = np.linspace(0, c, nz, endpoint=False)
    return z, vz


def smooth(y, window=11):
    """
    Moving average with edge padding to avoid boundary artifacts.
    """
    if window < 3:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def plateau_mean(arr, frac=0.4):
    """
    Average the top fraction of values in a region to estimate plateau level.
    """
    m = max(3, int(len(arr) * frac))
    idx = np.argsort(arr)[-m:]
    return arr[idx].mean()


def estimate_vacuum_lr_and_mid(
    z,
    vz,
    smooth_window=11,
    end_fraction=0.18,
    edge_trim=0.03,
    mid_fraction=0.15,
    top_fraction=0.4
):
    """
    Estimate:
      - left vacuum level
      - right vacuum level
      - slab middle average potential
    """
    vs = smooth(vz, smooth_window)
    n = len(vs)

    n_trim = max(1, int(n * edge_trim))
    n_end = max(8, int(n * end_fraction))

    left = vs[n_trim:n_end]
    right = vs[n - n_end:n - n_trim]

    evac_left = plateau_mean(left, top_fraction)
    evac_right = plateau_mean(right, top_fraction)

    zmin = z.min()
    zmax = z.max()
    zc = 0.5 * (zmin + zmax)
    halfw = 0.5 * (zmax - zmin) * mid_fraction
    mask_mid = (z >= zc - halfw) & (z <= zc + halfw)
    vmid = vs[mask_mid].mean()

    return evac_left, evac_right, vmid, vs


def read_doscar_total(doscar="DOSCAR"):
    """
    Read total DOS from DOSCAR.
    Returns:
        egrid, dos, intdos, efermi, nedos, natoms
    Works for non-spin-polarized total DOS.
    """
    with open(doscar, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0].split()[0])
    header = lines[5].split()
    emax, emin, nedos, efermi = float(header[0]), float(header[1]), int(header[2]), float(header[3])

    egrid = []
    dos = []
    intdos = []

    for i in range(6, 6 + nedos):
        vals = lines[i].split()
        egrid.append(float(vals[0]))
        dos.append(float(vals[1]))
        intdos.append(float(vals[2]))

    return np.array(egrid), np.array(dos), np.array(intdos), efermi, nedos, natoms


def read_doscar_atom_pdos_sum(doscar="DOSCAR", atom_indices=None):
    """
    Read summed PDOS for selected atoms from DOSCAR.
    Assumes ISPIN=1.
    Sums all projected columns for each selected atom.

    Parameters
    ----------
    atom_indices : list of 1-based atom indices

    Returns
    -------
    egrid : np.ndarray
    pdos_sum : np.ndarray
    """
    if atom_indices is None or len(atom_indices) == 0:
        raise ValueError("atom_indices is empty.")

    with open(doscar, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0].split()[0])
    header = lines[5].split()
    nedos = int(header[2])

    for a in atom_indices:
        if a < 1 or a > natoms:
            raise ValueError(f"Atom index {a} out of range. DOSCAR contains {natoms} atoms.")

    # total DOS block:
    # lines 6 ... 6+nedos-1
    # then for each atom:
    # one header line + nedos lines
    block0 = 6 + nedos

    egrid = None
    pdos_sum = None

    for atom in atom_indices:
        block_start = block0 + (atom - 1) * (nedos + 1) + 1

        energies = []
        atom_pdos = []

        for i in range(block_start, block_start + nedos):
            vals = [float(x) for x in lines[i].split()]
            energies.append(vals[0])
            atom_pdos.append(sum(vals[1:]))

        energies = np.array(energies)
        atom_pdos = np.array(atom_pdos)

        if egrid is None:
            egrid = energies
            pdos_sum = atom_pdos
        else:
            if not np.allclose(egrid, energies, atol=1e-8):
                raise ValueError(f"Energy grid mismatch in atom {atom} block.")
            pdos_sum += atom_pdos

    return egrid, pdos_sum


def export_aligned_dos(outname, egrid, dos, efermi, evac_left=None, evac_right=None, vmid=None):
    cols = [egrid, dos, egrid - efermi]
    names = ["E_raw(eV)", "DOS(states/eV)", "E_minus_Ef(eV)"]

    if evac_left is not None:
        cols.append(egrid - evac_left)
        names.append("E_minus_Evac_left(eV)")

    if evac_right is not None:
        cols.append(egrid - evac_right)
        names.append("E_minus_Evac_right(eV)")

    if vmid is not None:
        cols.append(egrid - vmid)
        names.append("E_minus_Vmid(eV)")

    arr = np.column_stack(cols)
    header = "  ".join(names)
    np.savetxt(outname, arr, header=header)
    print(f"Saved {outname}")


def export_aligned_pdos(outname, egrid, pdos, efermi, evac_left=None, evac_right=None, vmid=None):
    cols = [egrid, pdos, egrid - efermi]
    names = ["E_raw(eV)", "PDOS(states/eV)", "E_minus_Ef(eV)"]

    if evac_left is not None:
        cols.append(egrid - evac_left)
        names.append("E_minus_Evac_left(eV)")

    if evac_right is not None:
        cols.append(egrid - evac_right)
        names.append("E_minus_Evac_right(eV)")

    if vmid is not None:
        cols.append(egrid - vmid)
        names.append("E_minus_Vmid(eV)")

    arr = np.column_stack(cols)
    header = "  ".join(names)
    np.savetxt(outname, arr, header=header)
    print(f"Saved {outname}")


def main(workdir=".", atom_str=None):
    workdir = Path(workdir)

    locpot = workdir / "LOCPOT"
    doscar = workdir / "DOSCAR"

    lattice, counts, grid, pot = read_locpot(locpot)
    z, vz = planar_average_z(pot, lattice)
    evac_left, evac_right, vmid, vs = estimate_vacuum_lr_and_mid(z, vz)

    egrid, dos, intdos, efermi, nedos, natoms = read_doscar_total(doscar)

    phi_left = evac_left - efermi
    phi_right = evac_right - efermi

    print(f"Directory    : {workdir}")
    print(f"E_F          = {efermi:.6f} eV")
    print(f"E_vac_left   = {evac_left:.6f} eV")
    print(f"E_vac_right  = {evac_right:.6f} eV")
    print(f"V_mid        = {vmid:.6f} eV")
    print(f"Phi_left     = {phi_left:.6f} eV")
    print(f"Phi_right    = {phi_right:.6f} eV")
    print(f"E_F - V_mid  = {efermi - vmid:.6f} eV")
    print(f"N_atoms      = {natoms}")
    print(f"NEDOS        = {nedos}")

    export_aligned_dos(
        workdir / "dos_aligned.dat",
        egrid,
        dos,
        efermi,
        evac_left,
        evac_right,
        vmid
    )

    if atom_str is None:
        atom_str = input("Please input first-layer atom indices (e.g. 1-4,7,8): ").strip()

    atom_indices = parse_atom_list(atom_str)
    print(f"Selected first-layer atoms (1-based): {atom_indices}")

    egrid_pdos, pdos_sum = read_doscar_atom_pdos_sum(doscar, atom_indices)

    if not np.allclose(egrid, egrid_pdos, atol=1e-8):
        raise ValueError("Total DOS and PDOS energy grids do not match.")

    export_aligned_pdos(
        workdir / "firstlayer_pdos_aligned.dat",
        egrid_pdos,
        pdos_sum,
        efermi,
        evac_left,
        evac_right,
        vmid
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    axes[0].plot(z, vz, label="planar avg LOCPOT")
    axes[0].plot(z, vs, label="smoothed")
    axes[0].axhline(evac_left, linestyle="--", label=f"Evac_left = {evac_left:.3f} eV")
    axes[0].axhline(evac_right, linestyle="--", label=f"Evac_right = {evac_right:.3f} eV")
    axes[0].axhline(vmid, linestyle="--", label=f"Vmid = {vmid:.3f} eV")
    axes[0].set_xlabel("z (Angstrom)")
    axes[0].set_ylabel("Potential (eV)")
    axes[0].legend()

    axes[1].plot(egrid - efermi, dos, label="Total DOS: E - Ef")
    axes[1].plot(egrid - efermi, pdos_sum, label="First-layer PDOS: E - Ef")
    axes[1].axvline(0, linestyle="--")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].set_ylabel("DOS / PDOS")
    axes[1].legend()
    axes[1].set_xlim(-7, 7)
    axes[1].set_ylim(-1, 40)

    plt.tight_layout()
    plt.savefig(workdir / "alignment_check.png", dpi=200)
    print(f"Saved {workdir / 'alignment_check.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read LOCPOT and DOSCAR, export aligned DOS and first-layer PDOS."
    )
    parser.add_argument(
        "workdir",
        nargs="?",
        default=".",
        help="Working directory containing LOCPOT and DOSCAR"
    )
    parser.add_argument(
        "--atoms",
        type=str,
        default=None,
        help='First-layer atom indices, e.g. "1-4,7,8"'
    )

    args = parser.parse_args()
    main(args.workdir, args.atoms)
