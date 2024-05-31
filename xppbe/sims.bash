main_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


for arg in "$@"; do
    case $arg in
        --sims-path=*)
        sims_path="${arg#*=}"
        shift
        ;;
        --results-path=*)
        results_path="${arg#*=}/results"
        shift 
        ;;
        --pbj)
        pbj_flag="--pbj"
        shift
        ;;
        --mesh)
        mesh_flag="--mesh"
        shift
        ;;
        *)
        remaining_args+=("$arg")
        ;;
    esac
done


python "$main_dir/simulation.py" "$main_dir/simulation.yaml" "--results-path=${results_path%/results}"