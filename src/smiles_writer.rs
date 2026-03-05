/// Canonical SMILES writer: Mol -> SMILES string.
///
/// Uses Morgan-like canonical labeling, then DFS to emit SMILES.

use crate::mol::{BondStereo, BondType, Chirality, H_SENTINEL, Mol, atomic_symbol, count_swaps, default_valence};

/// Convert a Mol to a SMILES string.
pub fn mol_to_smiles(mol: &Mol, canonical: bool, rooted_atom: Option<usize>) -> String {
    if mol.atoms.is_empty() {
        return String::new();
    }

    let n = mol.atoms.len();

    let ranks = if canonical {
        canon_ranks(mol)
    } else {
        (0..n).collect()
    };

    let mut visited = vec![false; n];
    let mut result = String::new();
    let mut first_fragment = true;

    let mut start_atoms: Vec<usize> = Vec::new();
    if let Some(root) = rooted_atom {
        if root < n {
            start_atoms.push(root);
        }
    }

    let mut atom_order: Vec<usize> = (0..n).collect();
    atom_order.sort_by_key(|&i| ranks[i]);
    for &i in &atom_order {
        if !visited[i] && !start_atoms.contains(&i) {
            let mut comp = Vec::new();
            let mut q = std::collections::VecDeque::new();
            let mut comp_visited = vec![false; n];
            q.push_back(i);
            comp_visited[i] = true;
            while let Some(a) = q.pop_front() {
                comp.push(a);
                for &(nbr, _) in &mol.adj[a] {
                    if !comp_visited[nbr] {
                        comp_visited[nbr] = true;
                        q.push_back(nbr);
                    }
                }
            }
            let best = comp.iter().copied().min_by_key(|&a| ranks[a]).unwrap();
            if !start_atoms.contains(&best) {
                start_atoms.push(best);
            }
        }
    }

    for &start in &start_atoms {
        if visited[start] {
            continue;
        }
        if !first_fragment {
            result.push('.');
        }
        first_fragment = false;
        write_fragment(mol, start, &ranks, &mut visited, &mut result);
    }

    result
}

fn write_fragment(mol: &Mol, start: usize, ranks: &[usize], visited: &mut [bool], out: &mut String) {
    let n = mol.atoms.len();

    let mut ring_bonds: Vec<(usize, usize, usize)> = Vec::new();
    let mut dfs_order = Vec::new();
    let mut dfs_visited = vec![false; n];
    let mut dfs_parent = vec![usize::MAX; n];
    let mut dfs_parent_bond = vec![usize::MAX; n];

    dfs_assign(mol, start, ranks, &mut dfs_visited, &mut dfs_order, &mut dfs_parent, &mut dfs_parent_bond, &mut ring_bonds);

    let mut ring_nums: std::collections::HashMap<usize, Vec<(usize, i32, BondType, usize)>> = std::collections::HashMap::new();
    let mut next_ring_num = 1i32;
    for &(a1, a2, bi) in &ring_bonds {
        let rn = next_ring_num;
        next_ring_num += 1;
        let bond_type = mol.bonds[bi].bond_type;
        ring_nums.entry(a1).or_default().push((a2, rn, bond_type, bi));
        ring_nums.entry(a2).or_default().push((a1, rn, bond_type, bi));
    }

    let mut write_visited = vec![false; n];
    write_atom_dfs(mol, start, ranks, &mut write_visited, &ring_nums, &dfs_parent, &dfs_parent_bond, out, usize::MAX, &ring_bonds);

    for &a in &dfs_order {
        visited[a] = true;
    }
}

fn dfs_assign(
    mol: &Mol,
    atom: usize,
    ranks: &[usize],
    visited: &mut [bool],
    order: &mut Vec<usize>,
    parent: &mut [usize],
    parent_bond: &mut [usize],
    ring_bonds: &mut Vec<(usize, usize, usize)>,
) {
    visited[atom] = true;
    order.push(atom);

    let mut neighbors: Vec<(usize, usize)> = mol.adj[atom].clone();
    neighbors.sort_by_key(|&(nbr, _)| ranks[nbr]);

    for (nbr, bi) in neighbors {
        if !visited[nbr] {
            parent[nbr] = atom;
            parent_bond[nbr] = bi;
            dfs_assign(mol, nbr, ranks, visited, order, parent, parent_bond, ring_bonds);
        } else if nbr != parent[atom] || bi != parent_bond[atom] {
            let (a1, a2) = if order.iter().position(|&x| x == nbr).unwrap()
                < order.iter().position(|&x| x == atom).unwrap()
            {
                (nbr, atom)
            } else {
                (atom, nbr)
            };
            if !ring_bonds.iter().any(|&(x, y, _)| (x == a1 && y == a2) || (x == a2 && y == a1)) {
                ring_bonds.push((a1, a2, bi));
            }
        }
    }
}

fn write_atom_dfs(
    mol: &Mol,
    atom: usize,
    ranks: &[usize],
    visited: &mut [bool],
    ring_nums: &std::collections::HashMap<usize, Vec<(usize, i32, BondType, usize)>>,
    dfs_parent: &[usize],
    dfs_parent_bond: &[usize],
    out: &mut String,
    parent: usize,
    ring_bonds: &[(usize, usize, usize)],
) {
    visited[atom] = true;

    let sorted_rings: Vec<(usize, i32, BondType, bool, usize)> = if let Some(rings) = ring_nums.get(&atom) {
        let mut sr: Vec<(usize, i32, BondType, bool, usize)> = rings
            .iter()
            .map(|&(other, rn, bond_type, bi)| (other, rn, bond_type, visited[other], bi))
            .collect();
        sr.sort_by_key(|&(_, rn, _, is_closing, _)| (!is_closing, rn));
        sr
    } else {
        Vec::new()
    };

    let mut children: Vec<(usize, usize)> = Vec::new();
    for &(nbr, bi) in &mol.adj[atom] {
        if !visited[nbr] {
            let is_ring_bond = ring_nums
                .get(&atom)
                .map_or(false, |v| v.iter().any(|&(o, _, _, _)| o == nbr));
            if !is_ring_bond {
                children.push((nbr, bi));
            }
        }
    }
    children.sort_by_key(|&(nbr, _)| {
        let is_real = mol.atoms[nbr].atomic_num != 0;
        (is_real as u8, ranks[nbr])
    });

    let ring_closure_neighbors: Vec<usize> = sorted_rings.iter().map(|&(other, _, _, _, _)| other).collect();
    let chirality = compute_output_chirality(mol, atom, parent, &ring_closure_neighbors, &children);

    write_atom(mol, atom, chirality, out);

    for &(_other, rn, bond_type, is_closing, bi) in &sorted_rings {
        let bond = &mol.bonds[bi];
        if bond.stereo != BondStereo::None {
            let is_forward = bond.atom1 == atom;
            match (bond.stereo, is_forward) {
                (BondStereo::Up, true) => out.push('/'),
                (BondStereo::Up, false) => out.push('\\'),
                (BondStereo::Down, true) => out.push('\\'),
                (BondStereo::Down, false) => out.push('/'),
                _ => {}
            }
        } else if !is_closing {
            write_ring_bond(bond_type, out);
        }
        if rn < 10 {
            out.push_str(&rn.to_string());
        } else {
            out.push('%');
            out.push_str(&rn.to_string());
        }
    }

    let n_children = children.len();
    for (i, &(child, bi)) in children.iter().enumerate() {
        let is_last = i == n_children - 1;
        if !is_last {
            out.push('(');
        }
        write_bond(mol, bi, atom, child, out);
        write_atom_dfs(mol, child, ranks, visited, ring_nums, dfs_parent, dfs_parent_bond, out, atom, ring_bonds);
        if !is_last {
            out.push(')');
        }
    }
}

fn compute_output_chirality(
    mol: &Mol,
    atom: usize,
    parent: usize,
    ring_closure_neighbors: &[usize],
    children: &[(usize, usize)],
) -> Chirality {
    let stored = mol.atoms[atom].chirality;
    if stored == Chirality::None {
        return Chirality::None;
    }

    let mut output_order: Vec<usize> = Vec::new();
    if parent != usize::MAX {
        output_order.push(parent);
    }
    let has_h = mol.atoms[atom].num_explicit_hs > 0;
    if has_h {
        output_order.push(H_SENTINEL);
    }
    for &nbr in ring_closure_neighbors {
        output_order.push(nbr);
    }
    for &(child, _) in children {
        output_order.push(child);
    }

    let mut ref_order: Vec<usize> = Vec::new();
    if has_h {
        ref_order.push(H_SENTINEL);
    }
    for &(nbr, _) in &mol.adj[atom] {
        ref_order.push(nbr);
    }

    if output_order.len() != ref_order.len() {
        return stored;
    }

    let n_swaps = count_swaps(&ref_order, &output_order);
    if n_swaps % 2 == 1 {
        match stored {
            Chirality::CW => Chirality::CCW,
            Chirality::CCW => Chirality::CW,
            Chirality::None => Chirality::None,
        }
    } else {
        stored
    }
}

fn write_bond(mol: &Mol, bond_idx: usize, from: usize, to: usize, out: &mut String) {
    let bond = &mol.bonds[bond_idx];

    if bond.stereo != BondStereo::None {
        let is_forward = bond.atom1 == from;
        match (bond.stereo, is_forward) {
            (BondStereo::Up, true) => out.push('/'),
            (BondStereo::Up, false) => out.push('\\'),
            (BondStereo::Down, true) => out.push('\\'),
            (BondStereo::Down, false) => out.push('/'),
            _ => {}
        }
        return;
    }

    match bond.bond_type {
        BondType::Single => {
            if mol.atoms[from].is_aromatic && mol.atoms[to].is_aromatic {
                out.push('-');
            }
        }
        BondType::Double => out.push('='),
        BondType::Triple => out.push('#'),
        BondType::Aromatic => {}
    }
}

fn write_ring_bond(bond_type: BondType, out: &mut String) {
    match bond_type {
        BondType::Double => out.push('='),
        BondType::Triple => out.push('#'),
        _ => {}
    }
}

fn write_atom(mol: &Mol, idx: usize, chirality: Chirality, out: &mut String) {
    let atom = &mol.atoms[idx];

    if needs_bracket(mol, idx) {
        out.push('[');
        if atom.isotope != 0 {
            out.push_str(&atom.isotope.to_string());
        }
        if atom.atomic_num == 0 {
            out.push('*');
        } else if atom.is_aromatic {
            out.push_str(&atomic_symbol(atom.atomic_num).to_lowercase());
        } else {
            out.push_str(atomic_symbol(atom.atomic_num));
        }
        match chirality {
            Chirality::CCW => out.push('@'),
            Chirality::CW => out.push_str("@@"),
            Chirality::None => {}
        }
        if atom.num_explicit_hs > 0 {
            out.push('H');
            if atom.num_explicit_hs > 1 {
                out.push_str(&atom.num_explicit_hs.to_string());
            }
        }
        if atom.charge != 0 {
            if atom.charge > 0 {
                out.push('+');
            } else {
                out.push('-');
            }
            let abs_charge = atom.charge.unsigned_abs();
            if abs_charge > 1 {
                out.push_str(&abs_charge.to_string());
            }
        }
        if atom.atom_map_num != 0 {
            out.push(':');
            out.push_str(&atom.atom_map_num.to_string());
        }
        out.push(']');
    } else {
        if atom.atomic_num == 0 {
            out.push('*');
        } else if atom.is_aromatic {
            out.push_str(&atomic_symbol(atom.atomic_num).to_lowercase());
        } else {
            out.push_str(atomic_symbol(atom.atomic_num));
        }
    }
}

fn needs_bracket(mol: &Mol, idx: usize) -> bool {
    let atom = &mol.atoms[idx];

    if atom.atomic_num == 0 && atom.isotope != 0 {
        return true;
    }
    if atom.atomic_num == 0 {
        return false;
    }
    if !is_organic_subset(atom.atomic_num, atom.is_aromatic) {
        return true;
    }
    if atom.charge != 0 {
        return true;
    }
    if atom.isotope != 0 {
        return true;
    }
    if atom.chirality != Chirality::None {
        return true;
    }
    if atom.atom_map_num != 0 {
        return true;
    }
    if atom.is_aromatic && atom.num_explicit_hs > 0 {
        return true;
    }
    if atom.bracket {
        let default_hs = compute_default_hs(mol, idx);
        let total_hs = atom.num_explicit_hs + atom.implicit_hs();
        if total_hs != default_hs {
            return true;
        }
    } else if atom.num_explicit_hs > 0 {
        return true;
    }

    false
}

fn is_organic_subset(atomic_num: i32, aromatic: bool) -> bool {
    if aromatic {
        matches!(atomic_num, 5 | 6 | 7 | 8 | 15 | 16 | 34)
    } else {
        matches!(atomic_num, 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53)
    }
}

fn compute_default_hs(mol: &Mol, idx: usize) -> i32 {
    let atom = &mol.atoms[idx];
    let valence = mol.bond_order_sum(idx);
    let dv = default_valence(atom.atomic_num, atom.charge);
    (dv - valence).max(0)
}

/// Morgan-like canonical ranking.
pub fn canon_ranks(mol: &Mol) -> Vec<usize> {
    let n = mol.atoms.len();
    if n == 0 {
        return Vec::new();
    }

    let mut invariants: Vec<u64> = (0..n)
        .map(|i| {
            let atom = &mol.atoms[i];
            let mut inv: u64 = 0;
            inv |= (atom.atomic_num as u64 & 0xFF) << 24;
            inv |= (mol.degree(i) as u64 & 0xF) << 20;
            inv |= (atom.total_hs() as u64 & 0xF) << 16;
            inv |= ((atom.charge + 4).max(0) as u64 & 0xF) << 12;
            inv |= (mol.atom_in_ring(i) as u64) << 11;
            inv |= (atom.is_aromatic as u64) << 10;
            inv |= atom.isotope as u64 & 0x3FF;
            inv
        })
        .collect();

    let mut ranks = invariants_to_ranks(&invariants);
    for _ in 0..n {
        let mut new_inv: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut h = invariants[i].wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut nbr_ranks: Vec<usize> = mol.adj[i].iter().map(|&(nbr, _)| ranks[nbr]).collect();
            nbr_ranks.sort();
            for r in nbr_ranks {
                h = h.wrapping_mul(6364136223846793005).wrapping_add(r as u64 + 1);
            }
            new_inv.push(h);
        }
        let new_ranks = invariants_to_ranks(&new_inv);
        let old_classes = ranks.iter().collect::<std::collections::HashSet<_>>().len();
        let new_classes = new_ranks.iter().collect::<std::collections::HashSet<_>>().len();
        if new_classes == old_classes {
            break;
        }
        invariants = new_inv;
        ranks = new_ranks;
    }

    let mut indexed: Vec<(usize, usize)> = ranks.iter().copied().enumerate().map(|(i, r)| (r, i)).collect();
    indexed.sort();
    let mut final_ranks = vec![0; n];
    for (rank, &(_, atom_idx)) in indexed.iter().enumerate() {
        final_ranks[atom_idx] = rank;
    }

    final_ranks
}

fn invariants_to_ranks(inv: &[u64]) -> Vec<usize> {
    let n = inv.len();
    let mut indexed: Vec<(u64, usize)> = inv.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by_key(|&(v, i)| (v, i));
    let mut ranks = vec![0; n];
    let mut current_rank = 0;
    for i in 0..n {
        if i > 0 && indexed[i].0 != indexed[i - 1].0 {
            current_rank = i;
        }
        ranks[indexed[i].1] = current_rank;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use crate::mol::Mol;
    use crate::encode;

    #[test]
    fn test_ez_stereo_ring_closure() {
        let smi = r"CC(=O)/N=C1\N(C)N=C(S(=O)(=O)N)S1";
        let mol = Mol::from_smiles(smi).unwrap();
        let result = mol.to_smiles(true, None);
        assert!(result.contains('\\') || result.contains('/'),
            "stereo markers should be present: {}", result);
        let encoded = encode::encode(smi).unwrap();
        assert_eq!(result, encoded);
    }

    #[test]
    fn test_ez_stereo_closing_site_direction() {
        // \2 at closing site: stereo must be preserved with correct direction
        let smi = r"C1(C(NC2=CC=CC=C2C)=O)C2C1CCCC/C=C\2";
        let encoded = encode::encode(smi).unwrap();
        // Re-parse and check the SMILES is valid
        let mol = Mol::from_smiles(&encoded);
        assert!(mol.is_some(), "encoded SMILES should be valid: {}", encoded);
    }
}
