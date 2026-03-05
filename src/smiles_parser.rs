/// SMILES parser: converts a SMILES string into a `Mol`.

use crate::mol::{Atom, BondStereo, BondType, Chirality, H_SENTINEL, Mol, atomic_number, count_swaps};

/// Parse a SMILES string into a Mol. Returns None if parsing fails.
pub fn parse_smiles(smiles: &str) -> Option<Mol> {
    let mut parser = SmilesParser::new(smiles);
    parser.parse()?;
    parser.normalize_chirality();
    let mut mol = parser.mol;
    if !mol.sanitize() {
        return None;
    }
    Some(mol)
}

struct SmilesParser {
    chars: Vec<char>,
    pos: usize,
    mol: Mol,
    stack: Vec<usize>,
    ring_closures: std::collections::HashMap<i32, (usize, Option<BondType>, BondStereo)>,
    pending_bond: Option<BondType>,
    pending_stereo: BondStereo,
    chiral_nbrs: Vec<Vec<usize>>,
    chiral_ring_placeholders: std::collections::HashMap<i32, (usize, usize)>,
}

impl SmilesParser {
    fn new(smiles: &str) -> Self {
        SmilesParser {
            chars: smiles.chars().collect(),
            pos: 0,
            mol: Mol::new(),
            stack: Vec::new(),
            ring_closures: std::collections::HashMap::new(),
            pending_bond: None,
            pending_stereo: BondStereo::None,
            chiral_nbrs: Vec::new(),
            chiral_ring_placeholders: std::collections::HashMap::new(),
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn parse(&mut self) -> Option<()> {
        while self.pos < self.chars.len() {
            let c = self.peek()?;
            match c {
                '[' => self.parse_bracket_atom()?,
                '(' => {
                    self.advance();
                    if let Some(&top) = self.stack.last() {
                        self.stack.push(top);
                    }
                }
                ')' => {
                    self.advance();
                    self.stack.pop();
                }
                '-' => {
                    self.advance();
                    self.pending_bond = Some(BondType::Single);
                }
                '=' => {
                    self.advance();
                    self.pending_bond = Some(BondType::Double);
                }
                '#' => {
                    self.advance();
                    self.pending_bond = Some(BondType::Triple);
                }
                ':' => {
                    self.advance();
                    self.pending_bond = Some(BondType::Aromatic);
                }
                '/' => {
                    self.advance();
                    self.pending_stereo = BondStereo::Up;
                    self.pending_bond = Some(BondType::Single);
                }
                '\\' => {
                    self.advance();
                    self.pending_stereo = BondStereo::Down;
                    self.pending_bond = Some(BondType::Single);
                }
                '%' => {
                    self.advance();
                    let d1 = self.advance()?.to_digit(10)? as i32;
                    let d2 = self.advance()?.to_digit(10)? as i32;
                    let ring_num = d1 * 10 + d2;
                    self.handle_ring_closure(ring_num)?;
                }
                '.' => {
                    self.advance();
                    self.stack.clear();
                    self.pending_bond = None;
                    self.pending_stereo = BondStereo::None;
                }
                '0'..='9' => {
                    self.advance();
                    let ring_num = c.to_digit(10).unwrap() as i32;
                    self.handle_ring_closure(ring_num)?;
                }
                _ => {
                    self.parse_organic_atom()?;
                }
            }
        }
        if self.stack.len() > 1 {
            return None;
        }
        if !self.ring_closures.is_empty() {
            return None;
        }
        Some(())
    }

    fn parse_organic_atom(&mut self) -> Option<()> {
        let c = self.advance()?;
        let (atomic_num, aromatic) = match c {
            'B' => {
                if self.peek() == Some('r') {
                    self.advance();
                    (35, false)
                } else {
                    (5, false)
                }
            }
            'C' => {
                if self.peek() == Some('l') {
                    self.advance();
                    (17, false)
                } else {
                    (6, false)
                }
            }
            'N' => (7, false),
            'O' => (8, false),
            'P' => (15, false),
            'S' => {
                if self.peek() == Some('i') {
                    self.advance();
                    (14, false)
                } else if self.peek() == Some('e') {
                    self.advance();
                    (34, false)
                } else {
                    (16, false)
                }
            }
            'F' => (9, false),
            'I' => (53, false),
            'c' => (6, true),
            'n' => (7, true),
            'o' => (8, true),
            's' => (16, true),
            'p' => (15, true),
            '*' => (0, false),
            _ => return None,
        };

        let mut atom = Atom::new(atomic_num);
        atom.is_aromatic = aromatic;
        atom.bracket = false;

        let idx = self.mol.add_atom(atom);
        self.chiral_nbrs.push(Vec::new());
        self.connect_to_previous(idx);
        self.stack.push(idx);

        Some(())
    }

    fn parse_bracket_atom(&mut self) -> Option<()> {
        self.advance(); // consume '['
        let mut atom = Atom::new(0);
        atom.bracket = true;

        // Isotope
        let mut isotope = 0i32;
        let mut has_isotope = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                isotope = isotope * 10 + c.to_digit(10).unwrap() as i32;
                has_isotope = true;
                self.advance();
            } else {
                break;
            }
        }
        if has_isotope {
            atom.isotope = isotope;
        }

        // Element symbol
        let c = self.peek()?;
        if c == '*' {
            self.advance();
            atom.atomic_num = 0;
        } else if c.is_ascii_uppercase() {
            self.advance();
            let mut sym = String::new();
            sym.push(c);
            if let Some(c2) = self.peek() {
                if c2.is_ascii_lowercase() && c2 != 'h' || (c2 == 'l' || c2 == 'r' || c2 == 'e' || c2 == 'i' || c2 == 'n' || c2 == 'a' || c2 == 'g' || c2 == 'u') {
                    let mut test = sym.clone();
                    test.push(c2);
                    if atomic_number(&test).is_some() {
                        sym.push(c2);
                        self.advance();
                    }
                }
            }
            atom.atomic_num = atomic_number(&sym)?;
        } else if c.is_ascii_lowercase() {
            self.advance();
            let sym = c.to_string();
            atom.atomic_num = atomic_number(&sym)?;
            atom.is_aromatic = true;
        } else {
            return None;
        }

        // Chirality
        if self.peek() == Some('@') {
            self.advance();
            if self.peek() == Some('@') {
                self.advance();
                atom.chirality = Chirality::CW;
            } else {
                atom.chirality = Chirality::CCW;
            }
        }

        // H count
        if self.peek() == Some('H') {
            self.advance();
            let mut h_count = 1i32;
            if let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    h_count = c.to_digit(10).unwrap() as i32;
                    self.advance();
                }
            }
            atom.num_explicit_hs = h_count;
            atom.num_implicit_hs = 0;
        } else {
            atom.num_implicit_hs = 0;
        }

        // Charge
        if let Some(c) = self.peek() {
            if c == '+' || c == '-' {
                self.advance();
                let sign = if c == '+' { 1 } else { -1 };
                let mut mag = 0i32;
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        mag = mag * 10 + d.to_digit(10).unwrap() as i32;
                        self.advance();
                    } else if d == c {
                        mag += 1;
                        self.advance();
                    } else {
                        break;
                    }
                }
                if mag == 0 {
                    mag = 1;
                }
                atom.charge = sign * mag;
            }
        }

        // Atom map number
        if self.peek() == Some(':') {
            self.advance();
            let mut map_num = 0i32;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    map_num = map_num * 10 + c.to_digit(10).unwrap() as i32;
                    self.advance();
                } else {
                    break;
                }
            }
            atom.atom_map_num = map_num;
        }

        if self.advance() != Some(']') {
            return None;
        }

        let idx = self.mol.add_atom(atom);
        self.chiral_nbrs.push(Vec::new());
        self.connect_to_previous(idx);
        if self.mol.atoms[idx].chirality != Chirality::None && self.mol.atoms[idx].num_explicit_hs > 0 {
            self.chiral_nbrs[idx].push(H_SENTINEL);
        }
        self.stack.push(idx);
        Some(())
    }

    fn connect_to_previous(&mut self, idx: usize) {
        if let Some(&prev) = self.stack.last() {
            let bond_type = self.resolve_bond_type(prev, idx);
            let bi = self.mol.add_bond(prev, idx, bond_type);
            self.mol.bonds[bi].stereo = self.pending_stereo;
            self.pending_bond = None;
            self.pending_stereo = BondStereo::None;
            if self.mol.atoms[prev].chirality != Chirality::None {
                self.chiral_nbrs[prev].push(idx);
            }
            if self.mol.atoms[idx].chirality != Chirality::None {
                self.chiral_nbrs[idx].push(prev);
            }
            self.stack.pop();
        } else {
            self.pending_bond = None;
            self.pending_stereo = BondStereo::None;
        }
    }

    fn resolve_bond_type(&self, a1: usize, a2: usize) -> BondType {
        if let Some(bt) = self.pending_bond {
            return bt;
        }
        if self.mol.atoms[a1].is_aromatic && self.mol.atoms[a2].is_aromatic {
            BondType::Aromatic
        } else {
            BondType::Single
        }
    }

    fn handle_ring_closure(&mut self, ring_num: i32) -> Option<()> {
        let current = *self.stack.last()?;

        if let Some((other, other_bt, other_stereo)) = self.ring_closures.remove(&ring_num) {
            let bond_type = if let Some(bt) = self.pending_bond {
                bt
            } else if let Some(bt) = other_bt {
                bt
            } else if self.mol.atoms[current].is_aromatic && self.mol.atoms[other].is_aromatic {
                BondType::Aromatic
            } else {
                BondType::Single
            };

            // Bond is stored as other→current (atom1=opening, atom2=closing).
            // Stereo from the opening site (other_stereo) is already in the
            // forward direction. Stereo from the closing site (pending_stereo)
            // describes direction from current, so flip to match other→current.
            let stereo = if self.pending_stereo != BondStereo::None {
                match self.pending_stereo {
                    BondStereo::Up => BondStereo::Down,
                    BondStereo::Down => BondStereo::Up,
                    BondStereo::None => BondStereo::None,
                }
            } else {
                other_stereo
            };

            let bi = self.mol.add_bond(other, current, bond_type);
            self.mol.bonds[bi].stereo = stereo;
            self.pending_bond = None;
            self.pending_stereo = BondStereo::None;

            if self.mol.atoms[current].chirality != Chirality::None {
                self.chiral_nbrs[current].push(other);
            }
            if let Some((_, pos)) = self.chiral_ring_placeholders.remove(&ring_num) {
                self.chiral_nbrs[other][pos] = current;
            }
        } else {
            self.ring_closures.insert(ring_num, (current, self.pending_bond, self.pending_stereo));
            self.pending_bond = None;
            self.pending_stereo = BondStereo::None;

            if self.mol.atoms[current].chirality != Chirality::None {
                let pos = self.chiral_nbrs[current].len();
                self.chiral_nbrs[current].push(usize::MAX - 1);
                self.chiral_ring_placeholders.insert(ring_num, (current, pos));
            }
        }

        Some(())
    }

    fn normalize_chirality(&mut self) {
        for i in 0..self.mol.atoms.len() {
            if self.mol.atoms[i].chirality == Chirality::None {
                continue;
            }
            let smiles_order = &self.chiral_nbrs[i];
            if smiles_order.is_empty() {
                continue;
            }

            let has_h = smiles_order.contains(&H_SENTINEL);
            let mut ref_order: Vec<usize> = Vec::new();
            if has_h {
                ref_order.push(H_SENTINEL);
            }
            for &(nbr, _) in &self.mol.adj[i] {
                ref_order.push(nbr);
            }

            if smiles_order.len() != ref_order.len() {
                continue;
            }

            let n_swaps = count_swaps(&ref_order, smiles_order);
            if n_swaps % 2 == 1 {
                self.mol.atoms[i].chirality = match self.mol.atoms[i].chirality {
                    Chirality::CW => Chirality::CCW,
                    Chirality::CCW => Chirality::CW,
                    Chirality::None => Chirality::None,
                };
            }
        }
    }
}
