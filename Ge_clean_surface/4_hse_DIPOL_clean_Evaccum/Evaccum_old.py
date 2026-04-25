import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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

    # species line and counts line
    # VASP 5 style: line 5 = symbols, line 6 = counts
    # VASP 4 style: line 5 = counts
    tokens5 = lines[5].split()
    if all(t.replace("-", "").isdigit() for t in tokens5):
        counts = [int(x) for x in lines[5].split()]
        coord_start = 7
    else:
        counts = [int(x) for x in lines[6].split()]
        coord_start = 8

    natoms = sum(counts)

    # possible "Selective dynamics"
    line = lines[coord_start - 1].strip().lower()
    if line.startswith("s"):
        coord_type_line = coord_start
        coord_start += 1
    else:
        coord_type_line = coord_start - 1

    # skip atom coordinates
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
    pot = vals.reshape((nx, ny, nz), order="F")  # VASP grid order

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


# def smooth(y, window=9):
#     if window < 3:
#         return y.copy()
#     kernel = np.ones(window) / window
#     return np.convolve(y, kernel, mode="same")

def smooth(y, window=11):
    if window < 3:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def estimate_vacuum_and_mid(z, vz, smooth_window=11, vacuum_fraction=0.15, mid_fraction=0.15):
    """
    Crude automatic estimator:
    - Evac: average of top vacuum_fraction highest-potential points after smoothing
    - Vmid: average of central mid_fraction region of the cell
    You should inspect the plot and adjust if needed.
    """
    vs = smooth(vz, smooth_window)

    # Evac from highest potential plateau-ish region
    n_vac = max(5, int(len(vs) * vacuum_fraction))
    top_idx = np.argsort(vs)[-n_vac:]
    evac = vs[top_idx].mean()

    # Vmid from central region of cell
    zmin = z.min()
    zmax = z.max()
    zc = 0.5 * (zmin + zmax)
    halfw = 0.5 * (zmax - zmin) * mid_fraction
    mask_mid = (z >= zc - halfw) & (z <= zc + halfw)
    vmid = vs[mask_mid].mean()

    return evac, vmid, vs


def read_doscar_total(doscar="DOSCAR"):
    """
    Read total DOS from DOSCAR.
    Returns:
        egrid, dos, intdos, efermi
    Works for non-spin-polarized total DOS.
    """
    with open(doscar, "r") as f:
        lines = f.readlines()

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

    return np.array(egrid), np.array(dos), np.array(intdos), efermi


def export_aligned_dos(outname, egrid, dos, efermi, evac=None, vmid=None):
    cols = [egrid, dos, egrid - efermi]
    names = ["E_raw(eV)", "DOS(states/eV)", "E_minus_Ef(eV)"]

    if evac is not None:
        cols.append(egrid - evac)
        names.append("E_minus_Evac(eV)")

    if vmid is not None:
        cols.append(egrid - vmid)
        names.append("E_minus_Vmid(eV)")

    arr = np.column_stack(cols)
    header = "  ".join(names)
    np.savetxt(outname, arr, header=header)
    print(f"Saved {outname}")


def main(workdir="."):
    workdir = Path(workdir)

    poscar = workdir / "POSCAR"
    locpot = workdir / "LOCPOT"
    doscar = workdir / "DOSCAR"

    lattice, counts, grid, pot = read_locpot(locpot)
    z, vz = planar_average_z(pot, lattice)
    evac, vmid, vs = estimate_vacuum_and_mid(z, vz)

    egrid, dos, intdos, efermi = read_doscar_total(doscar)

    print(f"Directory : {workdir}")
    print(f"E_F       = {efermi:.6f} eV")
    print(f"E_vac     = {evac:.6f} eV")
    print(f"V_mid     = {vmid:.6f} eV")
    print(f"Work func = E_vac - E_F = {evac - efermi:.6f} eV")
    print(f"E_F - E_vac = {efermi - evac:.6f} eV")
    print(f"E_F - V_mid = {efermi - vmid:.6f} eV")

    export_aligned_dos(workdir / "dos_aligned.dat", egrid, dos, efermi, evac, vmid)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    axes[0].plot(z, vz, label="planar avg LOCPOT")
    axes[0].plot(z, vs, label="smoothed")
    axes[0].axhline(evac, linestyle="--", label=f"Evac = {evac:.3f} eV")
    axes[0].axhline(vmid, linestyle="--", label=f"Vmid = {vmid:.3f} eV")
    axes[0].set_xlabel("z (Angstrom)")
    axes[0].set_ylabel("Potential (eV)")
    axes[0].legend()

    axes[1].plot(egrid - efermi, dos, label="E - Ef")
    axes[1].plot(egrid - evac, dos, label="E - Evac")
    axes[1].plot(egrid - vmid, dos, label="E - Vmid")
    axes[1].axvline(0, linestyle="--")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].set_ylabel("DOS")
    axes[1].legend()
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(workdir / "alignment_check.png", dpi=200)
    print(f"Saved {workdir / 'alignment_check.png'}")


if __name__ == "__main__":
    import sys
    wd = sys.argv[1] if len(sys.argv) > 1 else "."
    main(wd)
