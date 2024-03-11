# Download Guacamol molecules
wget https://ndownloader.figshare.com/files/13612760 -O raw-train.smiles
wget https://ndownloader.figshare.com/files/13612766 -O raw-valid.smiles
wget https://figshare.com/ndownloader/files/13612757 -O raw-test.smiles

# Filter them
for split in train valid test; do
    echo $split
    python process_smiles.py raw-${split}.smiles ${split}-all.smiles --inventory_csv ../eMolecules/emolecules_inventory.csv
done
