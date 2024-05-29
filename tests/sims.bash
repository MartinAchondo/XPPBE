main_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$main_dir"

python "$main_dir/sim.py"