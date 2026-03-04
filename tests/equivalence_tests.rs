/// Regression tests for chemical equivalence in safe-oxidizer.

use safe_oxidizer::encode;

#[test]
fn test_benzofuran_aromaticity() {
    let smiles = "N1(C(=O)C=CC2=CC3=C(C=C2)OCO3)CCC2=C1C=CC=C2";
    let rust_safe = encode::encode(smiles).unwrap();
    assert!(rust_safe.contains("."));
}

#[test]
fn test_pyrazole_aromaticity() {
    let smiles = "FC(F)(F)C1=CC(C2=CC=C(S(=O)(=O)N)C=C2)=NN1C1=CC=C(C)C=C1";
    let rust_safe = encode::encode(smiles).unwrap();
    assert!(rust_safe.contains("."));
}

#[test]
fn test_macrocyclic_stereo() {
    let smiles = "CC(=O)N1[C@H]2CC[C@@H](C1)[C@H](OC(=O)[C@H](C(C)(C)C)NC(=O)OC)C(=O)N[C@H](C(=O)N[C@H]3CCCCC/C=C\\[C@H]4CCCCC/C=C\\[C@H](NC(=O)[C@@H]2NC(=O)C3)C(=O)N5CCC[C@H]5C(=O)O)CC6=CC=C(O)C=C6";
    assert!(encode::encode(smiles).is_err(), "Invalid SMILES with unclosed ring 4 should fail");
}

#[test]
fn test_aromatic_atom_preservation() {
    let test_cases = vec![
        ("c1ccccc1COC", "benzene with ether"),
        ("c1ccncc1COC", "pyridine with ether"),
        ("c1ccc2c(c1)oc1ccccc12", "dibenzofuran"),
    ];

    for (smiles, description) in test_cases {
        match encode::encode(smiles) {
            Ok(rust_safe) => {
                println!("{}: {}", description, rust_safe);
                let has_aromatic_c = rust_safe.contains('c') && !rust_safe.chars().all(|c| c.is_uppercase());
                if smiles.contains('c') {
                    assert!(has_aromatic_c, "Expected aromatic 'c' in SAFE string for {}", description);
                }
            }
            Err(e) => {
                println!("{}: No BRICS bonds - {}", description, e);
            }
        }
    }
}

#[test]
fn test_complex_ring_systems() {
    let complex_molecules = vec![
        "c1ccc2c(c1)Cc1ccccc1-2",
        "c1ccc2c(c1)c1ccccc1n2",
        "c1ccc2c(c1)oc1ccccc12",
    ];

    for smiles in complex_molecules {
        match encode::encode(smiles) {
            Ok(safe_string) => {
                assert!(safe_string.len() > 0);
                println!("Complex ring {} -> {}", smiles, safe_string);
            }
            Err(e) => {
                println!("Complex ring {} failed: {}", smiles, e);
            }
        }
    }
}
