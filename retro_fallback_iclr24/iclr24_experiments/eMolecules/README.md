# eMolecules

Contains code to process and standardize the eMolecules inventory.

The data was sourced from this URL: <https://downloads.emolecules.com/free-extended/2023-09-01/>.
At the time of writing, this was the latest snapshot of the eMolecules inventory (with purchasing tiers available).
A download was made at 12:11 UK time on 2023-09-08 of the following files:

- `orders.tsv.gz`
- `quote_bb.tsv.gz`

To process these files and create the inventory, run the script:

```bash
python create_inventory.py
```

Note that by default this script will use all available CPUs to canonicalize the SMILES quickly.
