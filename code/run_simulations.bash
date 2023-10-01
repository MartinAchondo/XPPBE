#!/bin/bash

current_directory="$(pwd)"
sims_folder="Simulation_list"
num_simulations=$(ls "$current_directory/$sims_folder"/*.py | wc -l) 

if [ "$num_simulations" -eq 0 ]; then
  echo "No Python scripts found in $sims_folder. Exiting."
  exit 1
fi

bash Allclean

cd "$current_directory/$sims_folder"

sims_count=0

for script_name in *.py
do
    ((sims_count++))
    echo "########"
    echo ""
    base_name="${script_name%.py}"
    echo "Running Simulation $sims_count : $script_name"
    cp "$current_directory/$sims_folder/$script_name" "$current_directory/$script_name"
    python "$current_directory/$script_name"
    cp "$current_directory/$script_name" "$current_directory/results/$base_name/$script_name"
    rm "$current_directory/$script_name"
    echo "Ending Simulation $sims_count : $script_name"
    echo ""
    echo "########"

done
