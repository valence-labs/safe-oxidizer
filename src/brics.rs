/// BRICS bond finding using hardcoded predicate functions.
///
/// The 46 BRICS SMARTS patterns decompose into 15 unique atom query predicates
/// and 46 (Q_left, Q_right, bond_type) triples. We iterate all bonds and test
/// each endpoint against the query pairs.

use crate::mol::{BondType, Mol};

/// Find BRICS bonds in a molecule.
/// Returns pairs of atom indices (a1, a2) for each cleavable bond.
pub fn find_brics_bonds(mol: &Mol) -> Vec<(usize, usize)> {
    let mut result: Vec<(usize, usize)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (_bi, bond) in mol.bonds.iter().enumerate() {
        let a1 = bond.atom1;
        let a2 = bond.atom2;

        for &(ql, qr, bond_must_be_double) in BRICS_PAIRS {
            if bond_must_be_double {
                if bond.bond_type != BondType::Double {
                    continue;
                }
            } else {
                if bond.bond_type != BondType::Single && bond.bond_type != BondType::Aromatic {
                    continue;
                }
            }
            if bond.is_in_ring {
                continue;
            }

            let matches_fwd = ql(mol, a1) && qr(mol, a2);
            let matches_rev = ql(mol, a2) && qr(mol, a1);

            if matches_fwd || matches_rev {
                let key = (a1.min(a2), a1.max(a2));
                if seen.insert(key) {
                    result.push((a1, a2));
                }
                break;
            }
        }
    }

    result
}

type QueryFn = fn(&Mol, usize) -> bool;

static BRICS_PAIRS: &[(QueryFn, QueryFn, bool)] = &[
    (q_acyl, q_ether_o, false),
    (q_acyl, q_amine_n, false),
    (q_acyl, q_ring_amide_n, false),
    (q_ether_o, q_sp3_c_bonded_c, false),
    (q_ether_o, q_c_heterocycle, false),
    (q_ether_o, q_arom_c_hetero, false),
    (q_ether_o, q_c_carbocycle, false),
    (q_ether_o, q_arom_c_ring, false),
    (q_sp3_c_bonded_c, q_amine_n, false),
    (q_sp3_c_bonded_c, q_thioether_s, false),
    (q_amine_n, q_sulfonyl, false),
    (q_amine_n, q_arom_c_hetero, false),
    (q_amine_n, q_arom_c_ring, false),
    (q_amine_n, q_c_heterocycle, false),
    (q_amine_n, q_c_carbocycle, false),
    (q_acyl_notr, q_c_heterocycle, false),
    (q_acyl_notr, q_arom_c_hetero, false),
    (q_acyl_notr, q_c_carbocycle, false),
    (q_acyl_notr, q_arom_c_ring, false),
    (q_cc_d23, q_cc_d23, true),
    (q_c_notr_notd1, q_arom_n_neutral, false),
    (q_c_notr_notd1, q_ring_amide_n, false),
    (q_c_notr_notd1, q_c_heterocycle, false),
    (q_c_notr_notd1, q_arom_c_hetero, false),
    (q_c_notr_notd1, q_c_carbocycle, false),
    (q_c_notr_notd1, q_arom_c_ring, false),
    (q_arom_n_neutral, q_c_heterocycle, false),
    (q_arom_n_neutral, q_arom_c_hetero, false),
    (q_arom_n_neutral, q_c_carbocycle, false),
    (q_arom_n_neutral, q_arom_c_ring, false),
    (q_ring_amide_n, q_c_heterocycle, false),
    (q_ring_amide_n, q_arom_c_hetero, false),
    (q_ring_amide_n, q_c_carbocycle, false),
    (q_ring_amide_n, q_arom_c_ring, false),
    (q_thioether_s, q_c_heterocycle, false),
    (q_thioether_s, q_arom_c_hetero, false),
    (q_thioether_s, q_c_carbocycle, false),
    (q_thioether_s, q_arom_c_ring, false),
    (q_c_heterocycle, q_arom_c_hetero, false),
    (q_c_heterocycle, q_c_carbocycle, false),
    (q_c_heterocycle, q_arom_c_ring, false),
    (q_arom_c_hetero, q_arom_c_hetero, false),
    (q_arom_c_hetero, q_c_carbocycle, false),
    (q_arom_c_hetero, q_arom_c_ring, false),
    (q_c_carbocycle, q_arom_c_ring, false),
    (q_arom_c_ring, q_arom_c_ring, false),
];

fn is_heteroatom_or_c(anum: i32) -> bool {
    matches!(anum, 6 | 7 | 8 | 16)
}

fn is_heteroatom(anum: i32) -> bool {
    matches!(anum, 7 | 8 | 16)
}

fn has_double_bond(mol: &Mol, atom: usize) -> bool {
    mol.adj[atom]
        .iter()
        .any(|&(_, bi)| mol.bonds[bi].bond_type == BondType::Double)
}

fn has_ring_bond_to(mol: &Mol, atom: usize, pred: impl Fn(i32) -> bool) -> bool {
    mol.adj[atom].iter().any(|&(nbr, bi)| {
        mol.bonds[bi].is_in_ring && pred(mol.atoms[nbr].atomic_num)
    })
}

fn has_nonring_single_bond_to(mol: &Mol, atom: usize, pred: impl Fn(i32) -> bool) -> bool {
    mol.adj[atom].iter().any(|&(nbr, bi)| {
        mol.bonds[bi].bond_type == BondType::Single
            && !mol.bonds[bi].is_in_ring
            && pred(mol.atoms[nbr].atomic_num)
    })
}

fn has_double_bond_to(mol: &Mol, atom: usize, anum: i32) -> bool {
    mol.adj[atom].iter().any(|&(nbr, bi)| {
        mol.bonds[bi].bond_type == BondType::Double && mol.atoms[nbr].atomic_num == anum
    })
}

fn has_aromatic_bond_to(mol: &Mol, atom: usize, pred: impl Fn(i32, bool) -> bool) -> bool {
    mol.adj[atom].iter().any(|&(nbr, bi)| {
        mol.bonds[bi].bond_type == BondType::Aromatic
            && pred(mol.atoms[nbr].atomic_num, mol.atoms[nbr].is_aromatic)
    })
}

fn count_aromatic_bonds_to(mol: &Mol, atom: usize, pred: impl Fn(i32, bool) -> bool) -> usize {
    mol.adj[atom]
        .iter()
        .filter(|&&(nbr, bi)| {
            mol.bonds[bi].bond_type == BondType::Aromatic
                && pred(mol.atoms[nbr].atomic_num, mol.atoms[nbr].is_aromatic)
        })
        .count()
}

// ---- Query predicates ----

/// Acyl carbon: C, degree 3, neighbor in {*,C,N,O}, has =O
fn q_acyl(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    a.atomic_num == 6
        && !a.is_aromatic
        && mol.degree(atom) == 3
        && mol.adj[atom]
            .iter()
            .any(|&(nbr, _)| matches!(mol.atoms[nbr].atomic_num, 0 | 6 | 7 | 8))
        && has_double_bond_to(mol, atom, 8)
}

/// Ether/ester oxygen: O, degree 2, non-ring single bond to *,C,H
fn q_ether_o(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    a.atomic_num == 8
        && !a.is_aromatic
        && mol.degree(atom) == 2
        && has_nonring_single_bond_to(mol, atom, |anum| matches!(anum, 0 | 1 | 6))
}

/// Amine nitrogen: N, not terminal, no double bonds, neighbors are C/S/*/H, not ring amide
fn q_amine_n(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 7 || a.is_aromatic {
        return false;
    }
    if mol.degree(atom) <= 1 {
        return false;
    }
    if has_double_bond(mol, atom) {
        return false;
    }
    let all_nbrs_ok = mol.adj[atom].iter().all(|&(nbr, bi)| {
        if mol.bonds[bi].bond_type == BondType::Single {
            matches!(mol.atoms[nbr].atomic_num, 0 | 1 | 6 | 16)
        } else {
            true
        }
    });
    if !all_nbrs_ok {
        return false;
    }
    if mol.atom_in_ring(atom) {
        for &(nbr, bi) in &mol.adj[atom] {
            if mol.bonds[bi].is_in_ring
                && mol.atoms[nbr].atomic_num == 6
                && mol.atom_in_ring(nbr)
                && has_double_bond_to(mol, nbr, 8)
            {
                return false;
            }
        }
    }
    true
}

/// Ring amide nitrogen: ring N with ring bond to C(=O) and ring bond to C/N/O/S
fn q_ring_amide_n(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 7 || !mol.atom_in_ring(atom) {
        return false;
    }
    let has_co = mol.adj[atom].iter().any(|&(nbr, bi)| {
        mol.bonds[bi].is_in_ring
            && mol.atoms[nbr].atomic_num == 6
            && has_double_bond_to(mol, nbr, 8)
    });
    if !has_co {
        return false;
    }
    has_ring_bond_to(mol, atom, is_heteroatom_or_c)
}

/// sp3-like C bonded to another C: C, not terminal, no double bonds
fn q_sp3_c_bonded_c(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    a.atomic_num == 6
        && !a.is_aromatic
        && mol.degree(atom) > 1
        && !has_double_bond(mol, atom)
        && has_nonring_single_bond_to(mol, atom, |anum| anum == 6)
}

/// C in heterocyclic ring: aliphatic C with ring bonds to {C,N,O,S} and {N,O,S}
fn q_c_heterocycle(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || a.is_aromatic {
        return false;
    }
    has_ring_bond_to(mol, atom, is_heteroatom_or_c)
        && has_ring_bond_to(mol, atom, is_heteroatom)
}

/// Aromatic C bonded to heteroatom: c with aromatic bonds to {c,n,o,s} and {n,o,s}
fn q_arom_c_hetero(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || !a.is_aromatic {
        return false;
    }
    let has_any_arom = has_aromatic_bond_to(mol, atom, |anum, arom| {
        arom && matches!(anum, 6 | 7 | 8 | 16)
    });
    let has_hetero_arom = has_aromatic_bond_to(mol, atom, |anum, arom| {
        arom && matches!(anum, 7 | 8 | 16)
    });
    has_any_arom && has_hetero_arom
}

/// C in carbocyclic ring: aliphatic C with at least 2 ring bonds to C
fn q_c_carbocycle(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || a.is_aromatic {
        return false;
    }
    let ring_c_count = mol.adj[atom]
        .iter()
        .filter(|&&(nbr, bi)| mol.bonds[bi].is_in_ring && mol.atoms[nbr].atomic_num == 6)
        .count();
    ring_c_count >= 2
}

/// Aromatic C in all-C ring: c with aromatic bonds to at least 2 aromatic c
fn q_arom_c_ring(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || !a.is_aromatic {
        return false;
    }
    count_aromatic_bonds_to(mol, atom, |anum, arom| arom && anum == 6) >= 2
}

/// Thioether sulfur: S, degree 2, non-ring single bond to * or C
fn q_thioether_s(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    a.atomic_num == 16
        && !a.is_aromatic
        && mol.degree(atom) == 2
        && has_nonring_single_bond_to(mol, atom, |anum| matches!(anum, 0 | 6))
}

/// Sulfonyl sulfur: S, degree 4, neighbor C or *, two =O neighbors
fn q_sulfonyl(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 16 || a.is_aromatic {
        return false;
    }
    if mol.degree(atom) != 4 {
        return false;
    }
    let has_c_or_star = mol.adj[atom]
        .iter()
        .any(|&(nbr, _)| matches!(mol.atoms[nbr].atomic_num, 0 | 6));
    let double_o_count = mol.adj[atom]
        .iter()
        .filter(|&&(nbr, bi)| {
            mol.bonds[bi].bond_type == BondType::Double && mol.atoms[nbr].atomic_num == 8
        })
        .count();
    has_c_or_star && double_o_count >= 2
}

/// Acyl C not in ring: C, degree 3, not in ring, has =O, non-ring bond to *,C,N,O
fn q_acyl_notr(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    a.atomic_num == 6
        && !a.is_aromatic
        && mol.degree(atom) == 3
        && !mol.atom_in_ring(atom)
        && has_double_bond_to(mol, atom, 8)
        && has_nonring_single_bond_to(mol, atom, |anum| matches!(anum, 0 | 6 | 7 | 8))
}

/// C=C double bond pattern: C bonded to C, degree 2 or 3
fn q_cc_d23(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || a.is_aromatic {
        return false;
    }
    let d = mol.degree(atom);
    if d != 2 && d != 3 {
        return false;
    }
    mol.adj[atom]
        .iter()
        .any(|&(nbr, _)| mol.atoms[nbr].atomic_num == 6 && !mol.atoms[nbr].is_aromatic)
}

/// C not in ring, not terminal, all bonds single
fn q_c_notr_notd1(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 6 || a.is_aromatic {
        return false;
    }
    if mol.atom_in_ring(atom) || mol.degree(atom) <= 1 {
        return false;
    }
    mol.adj[atom]
        .iter()
        .all(|&(_, bi)| mol.bonds[bi].bond_type == BondType::Single)
}

/// Aromatic N, neutral, in ring with at least 2 aromatic bonds to {c,n,o,s}
fn q_arom_n_neutral(mol: &Mol, atom: usize) -> bool {
    let a = &mol.atoms[atom];
    if a.atomic_num != 7 || !a.is_aromatic {
        return false;
    }
    if a.charge != 0 {
        return false;
    }
    count_aromatic_bonds_to(mol, atom, |anum, arom| {
        arom && matches!(anum, 6 | 7 | 8 | 16)
    }) >= 2
}
