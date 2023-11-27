#!/bin/sh

folder="methanol"

path="${folder}"
mkdir -p $path

tulio_path="martinachondo@10.31.29.10:pbj_dir/pbj/Simulations/methanol"

achom_path="C:\\Users\\achom\\Desktop\\Main\\Code\\Scientific-Computing\\Physics-Neural-Networks\\PINN\\code\\tulio\\${path}\\."


echo $achom_path
pscp -r $tulio_path $achom_path