# Downloads and processes FusionRetro datasets.
# Script should be run from *this* directory.
git clone https://github.com/SongtaoLiu0823/FusionRetro.git

# Process datasets
for name in "train" "valid" "test" ; do
    python process_dataset.py FusionRetro/${name}_dataset.json ./${name}_dataset
done

# Remove git repo
rm -rf FusionRetro
