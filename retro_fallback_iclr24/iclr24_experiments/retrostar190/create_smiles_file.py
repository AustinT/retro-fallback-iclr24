from syntheseus_retro_star_benchmark import get_190_hard_test_smiles

if __name__ == "__main__":
    with open("retrostar190.smiles", "wt") as f:
        f.write("\n".join(get_190_hard_test_smiles()))
    print("DONE!")
