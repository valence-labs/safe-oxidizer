use std::sync::LazyLock;

use regex::Regex;

use crate::mol::Mol;

#[derive(Debug)]
pub enum SafeError {
    InvalidSmiles(String),
    FragmentationError(String),
    EncodeError(String),
}

impl std::fmt::Display for SafeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeError::InvalidSmiles(s) => write!(f, "Invalid SMILES: {}", s),
            SafeError::FragmentationError(s) => write!(f, "Fragmentation error: {}", s),
            SafeError::EncodeError(s) => write!(f, "Encode error: {}", s),
        }
    }
}

impl std::error::Error for SafeError {}

static RE_BRACKET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[.*?\]").unwrap());
static RE_ATTACH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\d+\*\]").unwrap());
static RE_WRONG_ATTACH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(([\%\d]*)\)").unwrap());
static RE_RDKIT_SAFE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(([=\-@#/\\]{0,2})(%?\d{1,2})\)").unwrap());

/// Find existing ring-closure numbers in a SMILES string.
fn find_branch_numbers(smiles: &str) -> Vec<i32> {
    let cleaned = RE_BRACKET.replace_all(smiles, "");
    let chars: Vec<char> = cleaned.chars().collect();
    let mut branch_numbers = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '%' {
            if i + 2 < chars.len() && chars[i + 1].is_ascii_digit() && chars[i + 2].is_ascii_digit() {
                let n = (chars[i + 1].to_digit(10).unwrap() * 10
                    + chars[i + 2].to_digit(10).unwrap()) as i32;
                branch_numbers.push(n);
                i += 3;
            } else {
                i += 1;
            }
        } else if chars[i].is_ascii_digit() {
            branch_numbers.push(chars[i].to_digit(10).unwrap() as i32);
            i += 1;
        } else {
            i += 1;
        }
    }
    branch_numbers
}

/// Encode a SMILES string into SAFE representation.
pub fn encode(smiles: &str) -> Result<String, SafeError> {
    let branch_numbers = find_branch_numbers(smiles);

    let mut mol = Mol::from_smiles(smiles)
        .ok_or_else(|| SafeError::InvalidSmiles(smiles.to_string()))?;

    // Set isotope on existing dummy atoms
    let mut bond_map_id: i32 = 1;
    for i in 0..mol.num_atoms() {
        if mol.atom_atomic_num(i) == 0 {
            mol.set_atom_map_num(i, 0);
            mol.set_atom_isotope(i, bond_map_id);
            bond_map_id += 1;
        }
    }

    // Find BRICS bonds
    let matching_bonds = mol.find_brics_bonds();
    if matching_bonds.is_empty() {
        return Ok(mol.to_smiles(true, None));
    }

    // Get bond indices
    let mut bond_indices = Vec::with_capacity(matching_bonds.len());
    for &(i_a, i_b) in &matching_bonds {
        if let Some(bi) = mol.bond_idx(i_a, i_b) {
            bond_indices.push(bi);
        }
    }

    if bond_indices.is_empty() {
        return Err(SafeError::FragmentationError(
            "No valid bonds to cut".to_string(),
        ));
    }

    let dummy_labels: Vec<(i32, i32)> = (0..bond_indices.len() as i32)
        .map(|i| (i + bond_map_id, i + bond_map_id))
        .collect();

    let fragmented = mol
        .fragment_on_bonds(&bond_indices, &dummy_labels)
        .ok_or_else(|| SafeError::EncodeError("FragmentOnBonds failed".to_string()))?;

    let mut frags = fragmented.get_mol_frags();
    frags.sort_by(|a, b| b.num_atoms().cmp(&a.num_atoms()));

    let mut frags_str = Vec::with_capacity(frags.len());
    for frag in &frags {
        let mut root_atom = None;
        for i in 0..frag.num_atoms() {
            if frag.atom_atomic_num(i) != 0 {
                root_atom = Some(i);
                break;
            }
        }
        let smiles = frag.to_smiles(true, root_atom);
        frags_str.push(smiles);
    }

    let mut scaffold_str = frags_str.join(".");

    let scf_branch_num = {
        let mut nums = find_branch_numbers(&scaffold_str);
        nums.extend_from_slice(&branch_numbers);
        nums
    };

    let attach_pos: Vec<String> = {
        let mut positions: Vec<String> = RE_ATTACH
            .find_iter(&scaffold_str)
            .map(|m| m.as_str().to_string())
            .collect();
        positions.sort();
        positions.dedup();
        positions
    };

    let starting_num = if scf_branch_num.is_empty() {
        1
    } else {
        scf_branch_num.iter().max().unwrap() + 1
    };

    for (i, attach) in attach_pos.iter().enumerate() {
        let num = starting_num + i as i32;
        let val = if num < 10 {
            format!("{}", num)
        } else {
            format!("%{}", num)
        };
        scaffold_str = scaffold_str.replace(attach.as_str(), &val);
    }

    scaffold_str = RE_WRONG_ATTACH
        .replace_all(&scaffold_str, "$1")
        .to_string();

    scaffold_str = RE_RDKIT_SAFE
        .replace_all(&scaffold_str, "$1$2")
        .to_string();

    Ok(scaffold_str)
}
