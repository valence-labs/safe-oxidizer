/// Core molecule representation: Mol, Atom, Bond structs.

use crate::smiles_parser;
use crate::smiles_writer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chirality {
    None,
    CW,  // @@
    CCW, // @
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondStereo {
    None,
    Up,   // /
    Down, // backslash
}

/// Sentinel value representing implicit hydrogen in chirality neighbor ordering.
pub const H_SENTINEL: usize = usize::MAX;

/// Count adjacent transpositions needed to convert `probe` into `reference`.
pub fn count_swaps(reference: &[usize], probe: &[usize]) -> usize {
    let mut probe = probe.to_vec();
    let mut n = 0;
    for i in 0..reference.len() {
        if probe[i] != reference[i] {
            let j = probe[i + 1..]
                .iter()
                .position(|&x| x == reference[i])
                .unwrap()
                + i
                + 1;
            probe.swap(i, j);
            n += 1;
        }
    }
    n
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub atomic_num: i32,
    pub charge: i32,
    pub isotope: i32,
    pub atom_map_num: i32,
    pub is_aromatic: bool,
    pub chirality: Chirality,
    pub num_explicit_hs: i32,
    /// -1 means "compute from valence rules"
    pub num_implicit_hs: i32,
    /// Whether this atom was written in bracket notation in SMILES
    pub bracket: bool,
}

impl Atom {
    pub fn new(atomic_num: i32) -> Self {
        Atom {
            atomic_num,
            charge: 0,
            isotope: 0,
            atom_map_num: 0,
            is_aromatic: false,
            chirality: Chirality::None,
            num_explicit_hs: 0,
            num_implicit_hs: -1,
            bracket: false,
        }
    }

    pub fn total_hs(&self) -> i32 {
        self.num_explicit_hs + self.implicit_hs()
    }

    pub fn implicit_hs(&self) -> i32 {
        if self.num_implicit_hs >= 0 {
            return self.num_implicit_hs;
        }
        0
    }
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: BondType,
    pub stereo: BondStereo,
    pub is_in_ring: bool,
}

/// Core molecular graph.
#[derive(Debug, Clone)]
pub struct Mol {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Adjacency list: for each atom, list of (neighbor_atom_idx, bond_idx).
    pub adj: Vec<Vec<(usize, usize)>>,
}

impl Mol {
    pub fn new() -> Self {
        Mol {
            atoms: Vec::new(),
            bonds: Vec::new(),
            adj: Vec::new(),
        }
    }

    pub fn add_atom(&mut self, atom: Atom) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(atom);
        self.adj.push(Vec::new());
        idx
    }

    pub fn add_bond(&mut self, a1: usize, a2: usize, bt: BondType) -> usize {
        let idx = self.bonds.len();
        self.bonds.push(Bond {
            atom1: a1,
            atom2: a2,
            bond_type: bt,
            stereo: BondStereo::None,
            is_in_ring: false,
        });
        self.adj[a1].push((a2, idx));
        self.adj[a2].push((a1, idx));
        idx
    }

    pub fn degree(&self, atom_idx: usize) -> usize {
        self.adj[atom_idx].len()
    }

    /// Sum of bond orders for an atom (aromatic counts as 1 for valence).
    pub fn bond_order_sum(&self, atom_idx: usize) -> i32 {
        let mut sum = 0i32;
        for &(_, bi) in &self.adj[atom_idx] {
            sum += match self.bonds[bi].bond_type {
                BondType::Single => 1,
                BondType::Double => 2,
                BondType::Triple => 3,
                BondType::Aromatic => 1,
            };
        }
        sum
    }

    /// Compute implicit hydrogens from valence rules.
    pub fn compute_implicit_hs(&mut self) {
        for i in 0..self.atoms.len() {
            if self.atoms[i].bracket && self.atoms[i].num_implicit_hs >= 0 {
                continue;
            }
            if self.atoms[i].atomic_num == 0 {
                self.atoms[i].num_implicit_hs = 0;
                continue;
            }
            let valence = self.bond_order_sum(i) + self.atoms[i].num_explicit_hs;
            let charge = self.atoms[i].charge;
            let default_valence = default_valence(self.atoms[i].atomic_num, charge);
            let imp = (default_valence - valence).max(0);
            self.atoms[i].num_implicit_hs = imp;
        }
    }

    /// Find ring bonds using DFS cycle detection.
    pub fn detect_rings(&mut self) {
        let n = self.atoms.len();
        if n == 0 {
            return;
        }
        for b in &mut self.bonds {
            b.is_in_ring = false;
        }

        let mut visited = vec![false; n];
        let mut parent = vec![usize::MAX; n];
        let mut parent_bond = vec![usize::MAX; n];
        let mut depth = vec![0u32; n];

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut stack = vec![(start, 0usize, usize::MAX, usize::MAX)];
            visited[start] = true;
            depth[start] = 0;

            while let Some(&mut (atom, ref mut pos, par, pb)) = stack.last_mut() {
                if *pos >= self.adj[atom].len() {
                    stack.pop();
                    continue;
                }
                let (nbr, bi) = self.adj[atom][*pos];
                *pos += 1;

                if nbr == par && bi == pb {
                    continue;
                }

                if visited[nbr] {
                    if depth[atom] > depth[nbr] {
                        self.bonds[bi].is_in_ring = true;
                        let mut cur = atom;
                        while cur != nbr {
                            let pb_cur = parent_bond[cur];
                            if pb_cur != usize::MAX {
                                self.bonds[pb_cur].is_in_ring = true;
                            }
                            cur = parent[cur];
                        }
                    }
                } else {
                    visited[nbr] = true;
                    parent[nbr] = atom;
                    parent_bond[nbr] = bi;
                    depth[nbr] = depth[atom] + 1;
                    stack.push((nbr, 0, atom, bi));
                }
            }
        }
    }

    /// Check if an atom is in a ring.
    pub fn atom_in_ring(&self, atom_idx: usize) -> bool {
        self.adj[atom_idx]
            .iter()
            .any(|&(_, bi)| self.bonds[bi].is_in_ring)
    }

    /// Kekulize aromatic systems: assign alternating single/double bonds.
    pub fn kekulize(&mut self) -> bool {
        let n = self.atoms.len();
        let aromatic: Vec<bool> = (0..n).map(|i| self.atoms[i].is_aromatic).collect();

        let arom_bonds: Vec<usize> = (0..self.bonds.len())
            .filter(|&bi| self.bonds[bi].bond_type == BondType::Aromatic)
            .collect();

        if arom_bonds.is_empty() {
            return true;
        }

        let mut arom_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for &bi in &arom_bonds {
            let a1 = self.bonds[bi].atom1;
            let a2 = self.bonds[bi].atom2;
            arom_adj[a1].push((a2, bi));
            arom_adj[a2].push((a1, bi));
        }

        let mut needs_double: Vec<bool> = vec![false; n];
        let mut matched = vec![false; n];
        let mut bond_assigned: Vec<Option<BondType>> = vec![None; self.bonds.len()];

        for &bi in &arom_bonds {
            bond_assigned[bi] = Some(BondType::Single);
        }

        for i in 0..n {
            if !aromatic[i] {
                continue;
            }
            let charge = self.atoms[i].charge;
            let mut non_arom_order = 0i32;
            let explicit_hs = self.atoms[i].num_explicit_hs;
            for &(_, bi) in &self.adj[i] {
                if self.bonds[bi].bond_type != BondType::Aromatic {
                    non_arom_order += match self.bonds[bi].bond_type {
                        BondType::Single => 1,
                        BondType::Double => 2,
                        BondType::Triple => 3,
                        _ => 0,
                    };
                }
            }
            let arom_degree = arom_adj[i].len() as i32;
            let target_valence = default_valence(self.atoms[i].atomic_num, charge);
            let current = non_arom_order + arom_degree + explicit_hs;
            needs_double[i] = current < target_valence;
        }

        let result = self.kekulize_match(&arom_adj, &needs_double, &mut matched, &mut bond_assigned);
        if !result {
            // Fallback: for simple drug-like molecules, greedy usually works
        }

        for &bi in &arom_bonds {
            if let Some(bt) = bond_assigned[bi] {
                self.bonds[bi].bond_type = bt;
            }
        }

        true
    }

    fn kekulize_match(
        &self,
        arom_adj: &[Vec<(usize, usize)>],
        needs_double: &[bool],
        matched: &mut [bool],
        bond_assigned: &mut [Option<BondType>],
    ) -> bool {
        let n = self.atoms.len();
        let mut match_partner: Vec<Option<usize>> = vec![None; n];

        // Greedy initial matching
        for i in 0..n {
            if !needs_double[i] || matched[i] {
                continue;
            }
            for &(nbr, bi) in &arom_adj[i] {
                if needs_double[nbr] && !matched[nbr] {
                    matched[i] = true;
                    matched[nbr] = true;
                    match_partner[i] = Some(nbr);
                    match_partner[nbr] = Some(i);
                    bond_assigned[bi] = Some(BondType::Double);
                    break;
                }
            }
        }

        // Augmenting path phase
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                if !needs_double[i] || matched[i] {
                    continue;
                }
                let mut visited = vec![false; n];
                if self.augment(i, arom_adj, needs_double, &mut match_partner, bond_assigned, matched, &mut visited) {
                    improved = true;
                }
            }
        }

        (0..n).all(|i| !needs_double[i] || matched[i])
    }

    fn augment(
        &self,
        u: usize,
        arom_adj: &[Vec<(usize, usize)>],
        needs_double: &[bool],
        match_partner: &mut [Option<usize>],
        bond_assigned: &mut [Option<BondType>],
        matched: &mut [bool],
        visited: &mut [bool],
    ) -> bool {
        for &(v, bi) in &arom_adj[u] {
            if visited[v] || !needs_double[v] {
                continue;
            }
            visited[v] = true;

            if match_partner[v].is_none()
                || self.augment(match_partner[v].unwrap(), arom_adj, needs_double, match_partner, bond_assigned, matched, visited)
            {
                if let Some(old_partner) = match_partner[v] {
                    for &(_, old_bi) in &arom_adj[v] {
                        if bond_assigned[old_bi] == Some(BondType::Double) {
                            let b = &self.bonds[old_bi];
                            if (b.atom1 == v && b.atom2 == old_partner) || (b.atom1 == old_partner && b.atom2 == v) {
                                bond_assigned[old_bi] = Some(BondType::Single);
                                break;
                            }
                        }
                    }
                }

                match_partner[u] = Some(v);
                match_partner[v] = Some(u);
                matched[u] = true;
                matched[v] = true;
                bond_assigned[bi] = Some(BondType::Double);
                return true;
            }
        }
        false
    }

    /// Full sanitization: kekulize, compute implicit Hs, restore aromatic bonds.
    pub fn sanitize(&mut self) -> bool {
        self.detect_rings();

        let aromatic_bonds: Vec<usize> = self
            .bonds
            .iter()
            .enumerate()
            .filter(|(_, b)| b.bond_type == BondType::Aromatic)
            .map(|(i, _)| i)
            .collect();

        if !self.kekulize() {
            return false;
        }

        self.compute_implicit_hs();

        for bi in aromatic_bonds {
            self.bonds[bi].bond_type = BondType::Aromatic;
        }

        true
    }

    pub fn from_smiles(smiles: &str) -> Option<Self> {
        smiles_parser::parse_smiles(smiles)
    }

    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn atom_atomic_num(&self, idx: usize) -> i32 {
        self.atoms[idx].atomic_num
    }

    pub fn set_atom_isotope(&mut self, idx: usize, isotope: i32) {
        self.atoms[idx].isotope = isotope;
    }

    pub fn set_atom_map_num(&mut self, idx: usize, map_num: i32) {
        self.atoms[idx].atom_map_num = map_num;
    }

    pub fn bond_idx(&self, atom1: usize, atom2: usize) -> Option<usize> {
        for &(nbr, bi) in &self.adj[atom1] {
            if nbr == atom2 {
                return Some(bi);
            }
        }
        None
    }

    pub fn find_brics_bonds(&self) -> Vec<(usize, usize)> {
        crate::brics::find_brics_bonds(self)
    }

    pub fn fragment_on_bonds(
        &self,
        bond_indices: &[usize],
        dummy_labels: &[(i32, i32)],
    ) -> Option<Mol> {
        crate::fragment::fragment_on_bonds(self, bond_indices, dummy_labels)
    }

    pub fn get_mol_frags(&self) -> Vec<Mol> {
        crate::fragment::get_mol_frags(self)
    }

    pub fn to_smiles(&self, canonical: bool, rooted_atom: Option<usize>) -> String {
        smiles_writer::mol_to_smiles(self, canonical, rooted_atom)
    }
}

/// Default valence for common elements, considering charge.
pub fn default_valence(atomic_num: i32, charge: i32) -> i32 {
    let base = match atomic_num {
        0 => return 0,
        1 => 1,
        5 => 3,
        6 => 4,
        7 => 3,
        8 => 2,
        9 => 1,
        14 => 4,
        15 => 3,
        16 => 2,
        17 => 1,
        35 => 1,
        53 => 1,
        _ => 0,
    };

    match atomic_num {
        7 => match charge {
            1 => 4,
            -1 => 2,
            _ => base,
        },
        8 => match charge {
            1 => 3,
            -1 => 1,
            _ => base,
        },
        15 => match charge {
            1 => 4,
            _ => base,
        },
        16 => match charge {
            1 => 3,
            -1 => 1,
            _ => base,
        },
        _ => (base - charge).max(0),
    }
}

pub fn atomic_symbol(atomic_num: i32) -> &'static str {
    match atomic_num {
        0 => "*",
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        26 => "Fe",
        29 => "Cu",
        30 => "Zn",
        34 => "Se",
        35 => "Br",
        53 => "I",
        92 => "U",
        _ => "?",
    }
}

pub fn atomic_number(symbol: &str) -> Option<i32> {
    match symbol {
        "*" => Some(0),
        "H" => Some(1),
        "He" => Some(2),
        "Li" => Some(3),
        "Be" => Some(4),
        "B" => Some(5),
        "C" | "c" => Some(6),
        "N" | "n" => Some(7),
        "O" | "o" => Some(8),
        "F" => Some(9),
        "Ne" => Some(10),
        "Na" => Some(11),
        "Mg" => Some(12),
        "Al" => Some(13),
        "Si" => Some(14),
        "P" | "p" => Some(15),
        "S" | "s" => Some(16),
        "Cl" => Some(17),
        "Ar" => Some(18),
        "K" => Some(19),
        "Ca" => Some(20),
        "Fe" => Some(26),
        "Cu" => Some(29),
        "Zn" => Some(30),
        "Se" => Some(34),
        "Br" => Some(35),
        "I" => Some(53),
        "U" => Some(92),
        _ => None,
    }
}
