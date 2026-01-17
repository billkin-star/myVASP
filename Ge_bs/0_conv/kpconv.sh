#!/bin/bash
#Perform k-point convergence test
#Written by Growl1234, based on Sobereva's bash script for convergence test of k-point for CP2K.
 
template_file=KPOINTS   #Template file of present system
kpoint_list=XYZ.txt   #File containing k-point list
vasp_bin=$(which vasp_std)   #CP2K command
nproc_to_use=1   #Total number of CPU cores to use
recalc=0 #0: Keep old folder and files if exist  1: Always remove them and recalculate

if [ $recalc -eq 1 ] ; then
    echo "Running: rm -r encut_*"
    rm -r encut_*
fi
input_file=KPOINTS
output_file=vasp.out
output_file_2=OUTCAR
plot_file=KP.txt

nline=`wc -l $kpoint_list |cut -d ' ' -f 1`
echo "Number of tests: $nline"

#Prepare input files
for ((i = 1; i <= $nline; i++)) ; do
    kpthis=$(awk -v iline=$i 'NR==iline' $kpoint_list)
    work_dir=kp_${i}
    if [ ! -d $work_dir ] ; then
        mkdir $work_dir
    fi
    sed -e "s/TO_BE_TESTED/${kpthis}/g" $template_file > $work_dir/$input_file
    cp INCAR $work_dir/
    cp POSCAR $work_dir/
    cp POTCAR $work_dir/
done


#Run input files
for ((i = 1; i <= $nline; i++)) ; do
    work_dir=kp_${i}
    cd $work_dir
    if [ ! -e $output_file_2 ] ; then
       echo "Running $work_dir/$input_file"
       mpirun -np $nproc_to_use $vasp_bin > $output_file
    else
       echo "$work_dir/$output_file_2 has existed, skip calculation"
    fi
    cd ..
done

#Print energies
echo "# k-point vs total energy" > $plot_file
echo "# Date: $(date)" >> $plot_file
echo "# PWD: $PWD" >> $plot_file
echo -n "#   k-points |  Energy (Hartree)  |   delte E  " >> $plot_file
printf "\n" >> $plot_file
itime=0
for ((i = 1; i <= $nline; i++)) ; do
    work_dir=kp_${i}
    kpthis=$(awk -v iline=$i 'NR==iline' $kpoint_list)
    total_energy=$(grep -e 'sigma->0' $work_dir/$output_file_2 |tail -1 | awk '{print $7}')
    if (( $itime == 0 )); then
      printf "     %s  %18.10f               " "$kpthis" $total_energy >> $plot_file
    else
      E_var=$(awk "BEGIN {printf \"%.8f\", $total_energy - $E_last}")
      printf "     %s  %18.10f  %15.10f" "$kpthis" $total_energy $E_var >> $plot_file
    fi
    printf "\n" >> $plot_file
    E_last=$total_energy
    itime=$(($itime+1))
done

echo "If finished normally, now check $plot_file"

