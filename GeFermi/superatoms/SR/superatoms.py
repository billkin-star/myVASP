from ase.io import read, write
atoms = read("/home/mayn/myVASP/GeFermi/Ge.cif")

# 构建3×3×3超胞
atoms_super = atoms.repeat((3, 3, 3))  # 216个原子

# 掺杂
atoms_super[47].symbol = 'Ga'  # 替换第43个原子

# 保存
write("POSCAR_final", atoms_super, format='vasp', vasp5=True, direct=True)
