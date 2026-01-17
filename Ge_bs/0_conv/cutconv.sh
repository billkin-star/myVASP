#!/bin/bash
#Perform ENCUT convergence test
#Written by Growl1234, based on Sobereva's bash script for convergence test of CUTOFF for CP2K.
 
template_file=INCAR   #Template file of present system
vasp_bin=$(which vasp_std)  #CP2K command
nproc_to_use=1   #Total number of CPU cores to use
recalc=0 #0: Keep old folder and files if exist  1: Always remove them and recalculate
#encuts= "400 425 450 475 500 525 550"
encuts=$(seq 220 30 340) #Considered encut range and step

if [ $recalc -eq 1 ] ; then
    echo "Running: rm -r encut_*"
    rm -r encut_*
fi
input_file=INCAR
output_file=vasp.out
output_file_2=OUTCAR
plot_file=ENCUT.txt

#Prepare input files
for ii in $encuts ; do
    work_dir=ENCUT_${ii}eV
    if [ ! -d $work_dir ] ; then
        mkdir $work_dir
    fi
    sed -e "s/TO_BE_TESTED/${ii}/g" $template_file > $work_dir/$input_file
    cp KPOINTS $work_dir/
    cp POSCAR $work_dir/
    cp POTCAR $work_dir/
done


#Run input files
for ii in $encuts ; do
   work_dir=ENCUT_${ii}eV
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
echo "# Grid encut vs total energy" > $plot_file
echo "# Date: $(date)" >> $plot_file
echo "# PWD: $PWD" >> $plot_file
echo -n "#   ENCUT |  Energy (eV)  |   delte E  " >> $plot_file
printf "\n" >> $plot_file
itime=0
for ii in $encuts ; do
    work_dir=ENCUT_${ii}eV
    total_energy=$(grep -e 'sigma->0' $work_dir/$output_file_2 |tail -1 | awk '{print $7}')
    if (( $itime == 0 )); then
      printf "%8.0f  %15.8f               " $ii $total_energy >> $plot_file
    else
      E_var=$(awk "BEGIN {printf \"%.8f\", $total_energy - $E_last}")
      printf "%8.0f  %15.8f  %12.8f" $ii $total_energy $E_var >> $plot_file
    fi
    printf "\n" >> $plot_file
    E_last=$total_energy
    itime=$(($itime+1))
done

echo "If finished normally, now check $plot_file"

