for f in *.smiles.gz ; do
    echo "Unzipping $f"
    gunzip -k $f
done
echo "Done unzipping data"
