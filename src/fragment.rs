/// Bond fragmentation and connected component extraction.

use crate::mol::{Atom, Mol};

/// Fragment a molecule by cutting specified bonds and inserting dummy atoms.
pub fn fragment_on_bonds(
    mol: &Mol,
    bond_indices: &[usize],
    dummy_labels: &[(i32, i32)],
) -> Option<Mol> {
    let cut_map: std::collections::HashMap<usize, usize> = bond_indices
        .iter()
        .enumerate()
        .map(|(i, &bi)| (bi, i))
        .collect();

    let mut result = Mol::new();

    for atom in &mol.atoms {
        result.add_atom(atom.clone());
    }

    for (bi, bond) in mol.bonds.iter().enumerate() {
        if let Some(&cut_idx) = cut_map.get(&bi) {
            let (l1, l2) = dummy_labels[cut_idx];

            let mut dummy1 = Atom::new(0);
            dummy1.isotope = l1;
            dummy1.num_implicit_hs = 0;
            dummy1.bracket = true;
            let d1_idx = result.add_atom(dummy1);
            result.add_bond(bond.atom1, d1_idx, bond.bond_type);

            let mut dummy2 = Atom::new(0);
            dummy2.isotope = l2;
            dummy2.num_implicit_hs = 0;
            dummy2.bracket = true;
            let d2_idx = result.add_atom(dummy2);
            result.add_bond(bond.atom2, d2_idx, bond.bond_type);
        } else {
            let new_bi = result.add_bond(bond.atom1, bond.atom2, bond.bond_type);
            result.bonds[new_bi].stereo = bond.stereo;
            result.bonds[new_bi].is_in_ring = bond.is_in_ring;
        }
    }

    Some(result)
}

/// Extract connected components as separate Mol objects.
pub fn get_mol_frags(mol: &Mol) -> Vec<Mol> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    let mut component = vec![usize::MAX; n];
    let mut num_components = 0;

    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let comp_id = num_components;
        num_components += 1;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        component[start] = comp_id;
        while let Some(atom) = queue.pop_front() {
            for &(nbr, _) in &mol.adj[atom] {
                if component[nbr] == usize::MAX {
                    component[nbr] = comp_id;
                    queue.push_back(nbr);
                }
            }
        }
    }

    let mut frags = Vec::with_capacity(num_components);
    for comp_id in 0..num_components {
        let mut frag = Mol::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];

        for i in 0..n {
            if component[i] == comp_id {
                let new_idx = frag.add_atom(mol.atoms[i].clone());
                old_to_new[i] = Some(new_idx);
            }
        }

        for bond in &mol.bonds {
            if component[bond.atom1] == comp_id {
                if let (Some(new_a1), Some(new_a2)) =
                    (old_to_new[bond.atom1], old_to_new[bond.atom2])
                {
                    let bi = frag.add_bond(new_a1, new_a2, bond.bond_type);
                    frag.bonds[bi].stereo = bond.stereo;
                    frag.bonds[bi].is_in_ring = bond.is_in_ring;
                }
            }
        }

        frags.push(frag);
    }

    frags
}
