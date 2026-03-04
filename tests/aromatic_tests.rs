/// Tests for aromatic atom and bond preservation.

use safe_oxidizer::mol::{Mol, BondType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aromatic_atom_parsing() {
        let test_cases = vec![
            "c1ccccc1",
            "c1ccncc1",
            "c1cccn1",
            "c1cc2c(cc1)oco2",
        ];

        for smiles in test_cases {
            let mol = Mol::from_smiles(smiles).expect(&format!("Failed to parse {}", smiles));

            let has_aromatic = (0..mol.num_atoms()).any(|i| mol.atoms[i].is_aromatic);
            assert!(has_aromatic, "Expected aromatic atoms in {}", smiles);

            let canonical_smiles = mol.to_smiles(true, None);
            let reparsed = Mol::from_smiles(&canonical_smiles).expect("Failed to reparse");

            for i in 0..mol.num_atoms() {
                let orig_aromatic = mol.atoms[i].is_aromatic;
                let reparsed_aromatic = reparsed.atoms[i].is_aromatic;
                assert_eq!(orig_aromatic, reparsed_aromatic,
                    "Aromaticity mismatch for atom {} in {} -> {}", i, smiles, canonical_smiles);
            }
        }
    }

    #[test]
    fn test_aromatic_bond_preservation() {
        let smiles = "c1ccccc1";
        let mol = Mol::from_smiles(smiles).unwrap();

        let aromatic_bond_count = mol.bonds.iter()
            .filter(|b| b.bond_type == BondType::Aromatic)
            .count();

        assert!(aromatic_bond_count > 0, "Expected aromatic bonds in benzene");

        let mut mol_copy = mol.clone();
        mol_copy.sanitize();

        let aromatic_after_sanitize = mol_copy.bonds.iter()
            .filter(|b| b.bond_type == BondType::Aromatic)
            .count();

        assert_eq!(aromatic_bond_count, aromatic_after_sanitize,
            "Aromatic bonds should be preserved after sanitization");
    }

    #[test]
    fn test_mixed_aromatic_aliphatic() {
        let smiles = "c1ccccc1CCO";
        let mol = Mol::from_smiles(smiles).unwrap();

        let aromatic_count = (0..mol.num_atoms())
            .filter(|&i| mol.atoms[i].is_aromatic)
            .count();
        let aliphatic_count = (0..mol.num_atoms())
            .filter(|&i| !mol.atoms[i].is_aromatic)
            .count();

        assert!(aromatic_count > 0, "Should have aromatic atoms");
        assert!(aliphatic_count > 0, "Should have aliphatic atoms");

        let smiles_out = mol.to_smiles(true, None);
        assert!(smiles_out.contains('c'), "Should contain lowercase aromatic carbon");
        assert!(smiles_out.contains('C'), "Should contain uppercase aliphatic carbon");
    }

    #[test]
    fn test_heteroaromatic_atoms() {
        let smiles = "c1ccncc1";
        let mol = Mol::from_smiles(smiles).unwrap();

        let aromatic_n_count = (0..mol.num_atoms())
            .filter(|&i| {
                let atom = &mol.atoms[i];
                atom.atomic_num == 7 && atom.is_aromatic
            })
            .count();

        assert_eq!(aromatic_n_count, 1, "Should have one aromatic nitrogen");

        let smiles_out = mol.to_smiles(true, None);
        assert!(smiles_out.contains('n'), "Should contain aromatic nitrogen 'n'");
    }
}

#[cfg(test)]
mod brics_aromatic_tests {
    use safe_oxidizer::brics::find_brics_bonds;
    use safe_oxidizer::mol::Mol;

    #[test]
    fn test_biphenyl_has_brics_bond() {
        let mol = Mol::from_smiles("c1ccc(-c2ccccc2)cc1").unwrap();
        let bonds = find_brics_bonds(&mol);
        assert!(!bonds.is_empty(), "Biphenyl should have a BRICS bond at the inter-ring bond");
    }

    #[test]
    fn test_biphenyl_encoding() {
        let result = safe_oxidizer::encode::encode("c1ccc(-c2ccccc2)cc1");
        assert!(result.is_ok(), "Biphenyl should encode successfully: {:?}", result.err());
    }

    #[test]
    fn test_phenylpyridine_has_brics_bond() {
        let mol = Mol::from_smiles("c1ccc(-c2ccccn2)cc1").unwrap();
        let bonds = find_brics_bonds(&mol);
        assert!(!bonds.is_empty(), "Phenylpyridine should have a BRICS bond");
    }
}

#[cfg(test)]
mod invalid_smiles_tests {
    use safe_oxidizer::mol::Mol;

    #[test]
    fn test_unclosed_parenthesis_rejected() {
        assert!(Mol::from_smiles("C[C@H](C1=C(F)C=C(C").is_none(),
            "Unclosed parenthesis should be rejected");
    }

    #[test]
    fn test_unclosed_ring_rejected() {
        assert!(Mol::from_smiles("C1CCC").is_none(),
            "Unclosed ring closure should be rejected");
    }

    #[test]
    fn test_valid_smiles_still_accepted() {
        assert!(Mol::from_smiles("c1ccccc1").is_some());
        assert!(Mol::from_smiles("C(=O)O").is_some());
        assert!(Mol::from_smiles("CC(C)C").is_some());
    }
}
