import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_aligned_file(filename):
    """
    Read aligned DOS/PDOS data file.

    Expected columns:
    0: E_raw(eV)
    1: DOS or PDOS
    2: E_minus_Ef(eV)
    3: E_minus_Evac_left(eV)
    4: E_minus_Evac_right(eV)
    5: E_minus_Vmid(eV)   # optional

    Returns a dict.
    """
    data = np.loadtxt(filename)

    result = {
        "E_raw": data[:, 0],
        "Y": data[:, 1],
        "E_minus_Ef": data[:, 2],
    }

    if data.shape[1] >= 5:
        result["E_minus_Evac_left"] = data[:, 3]
        result["E_minus_Evac_right"] = data[:, 4]

        # 取较高真空能级对应的坐标：
        # E - Evac_high = min(E-Evac_left, E-Evac_right)
        result["E_minus_Evac_high"] = np.minimum(data[:, 3], data[:, 4])

    if data.shape[1] >= 6:
        result["E_minus_Vmid"] = data[:, 5]

    return result


def auto_ylim(x1, y1, x2, y2, xmin=-5, xmax=5, scale=1.1, floor=5):
    mask1 = (x1 >= xmin) & (x1 <= xmax)
    mask2 = (x2 >= xmin) & (x2 <= xmax)

    ymax1 = np.max(y1[mask1]) if np.any(mask1) else 0.0
    ymax2 = np.max(y2[mask2]) if np.any(mask2) else 0.0

    ymax = max(ymax1, ymax2)
    ymax = max(ymax * scale, floor)
    return ymax


def plot_compare(x1, y1, x2, y2,
                 label1="clean", label2="dimer",
                 xlabel="Energy (eV)", ylabel="DOS",
                 title="Comparison",
                 xlim=(-5, 5), ylim=None,
                 outfile="plot.png"):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x1, y1, linewidth=2, label=label1)
    ax.plot(x2, y2, linewidth=2, label=label2)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile}")


def main():
    # ======== 文件路径 ========
    # 总 DOS 对齐文件
    clean_dos_file = "dos_aligned_clean.dat"
    dimer_dos_file = "dos_aligned_dimer.dat"

    # 第一层 PDOS 对齐文件
    clean_pdos_file = "firstlayer_pdos_aligned_clean.dat"
    dimer_pdos_file = "firstlayer_pdos_aligned_dimer.dat"

    # ======== 读取总 DOS ========
    clean_dos = read_aligned_file(clean_dos_file)
    dimer_dos = read_aligned_file(dimer_dos_file)

    # ---------- 图1: DOS, E - Ef ----------
    x1 = clean_dos["E_minus_Ef"]
    y1 = clean_dos["Y"] / 18
    x2 = dimer_dos["E_minus_Ef"]
    y2 = dimer_dos["Y"] / 36

    ymax_ef = auto_ylim(x1, y1, x2, y2, xmin=-5, xmax=5, scale=1.1, floor=2)

    plot_compare(
        x1, y1, x2, y2,
        label1="clean",
        label2="dimer",
        xlabel=r"$E - E_F$ (eV)",
        ylabel="DOS/atom",
        title="DOS aligned to Fermi level",
        xlim=(-5, 5),
        ylim=(0, ymax_ef),
        outfile="dos_compare_Ef.png"
    )

    # ---------- 图2: DOS, E - Evac_low ----------
    if "E_minus_Evac_high" not in clean_dos or "E_minus_Evac_high" not in dimer_dos:
        raise ValueError("Aligned DOS file does not contain Evac_left / Evac_right columns.")

    x1 = clean_dos["E_minus_Evac_high"]
    y1 = clean_dos["Y"] / 18
    x2 = dimer_dos["E_minus_Evac_high"]
    y2 = dimer_dos["Y"] / 36

    ymax_evac = auto_ylim(x1, y1, x2, y2, xmin=-8, xmax=2, scale=1.1, floor=2)

    plot_compare(
        x1, y1, x2, y2,
        label1="clean",
        label2="dimer",
        xlabel=r"$E - E_{\mathrm{vac,high}}$ (eV)",
        ylabel="DOS/atom",
        title="DOS aligned to higher vacuum level",
        xlim=(-8, 2),
        ylim=(0, ymax_evac),
        outfile="dos_compare_Evac_high.png"
    )

    # ======== 读取第一层 PDOS ========
    clean_pdos = read_aligned_file(clean_pdos_file)
    dimer_pdos = read_aligned_file(dimer_pdos_file)

    # ---------- 图3: PDOS, E - Ef ----------
    x1 = clean_pdos["E_minus_Ef"]
    y1 = clean_pdos["Y"] / 2
    x2 = dimer_pdos["E_minus_Ef"]
    y2 = dimer_pdos["Y"] / 4

    ymax_pdos_ef = auto_ylim(x1, y1, x2, y2, xmin=-5, xmax=5, scale=1.1, floor=1.5)

    plot_compare(
        x1, y1, x2, y2,
        label1="clean first-layer PDOS",
        label2="dimer first-layer PDOS",
        xlabel=r"$E - E_F$ (eV)",
        ylabel="PDOS/atom",
        title="First-layer PDOS aligned to Fermi level",
        xlim=(-5, 5),
        ylim=(0, ymax_pdos_ef),
        outfile="pdos_compare_Ef.png"
    )

    # ---------- 图4: PDOS, E - Evac_low ----------
    if "E_minus_Evac_high" not in clean_pdos or "E_minus_Evac_high" not in dimer_pdos:
        raise ValueError("Aligned PDOS file does not contain Evac_left / Evac_right columns.")

    x1 = clean_pdos["E_minus_Evac_high"]
    y1 = clean_pdos["Y"] / 2
    x2 = dimer_pdos["E_minus_Evac_high"]
    y2 = dimer_pdos["Y"] / 4

    ymax_pdos_evac = auto_ylim(x1, y1, x2, y2, xmin=-8, xmax=2, scale=1.1, floor=1.5)

    plot_compare(
        x1, y1, x2, y2,
        label1="clean first-layer PDOS",
        label2="dimer first-layer PDOS",
        xlabel=r"$E - E_{\mathrm{vac,high}}$ (eV)",
        ylabel="PDOS/atom",
        title="First-layer PDOS aligned to higher vacuum level",
        xlim=(-8, 2),
        ylim=(0, ymax_pdos_evac),
        outfile="pdos_compare_Evac_high.png"
    )


if __name__ == "__main__":
    main()