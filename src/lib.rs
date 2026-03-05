pub mod brics;
pub mod encode;
pub mod fragment;
pub mod mol;
pub mod smiles_parser;
pub mod smiles_writer;
pub mod tokenizer;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Encode a SMILES string to SAFE representation using BRICS fragmentation.
#[pyfunction]
fn safe_encode(smiles: &str) -> PyResult<String> {
    encode::encode(smiles).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Encode a batch of SMILES strings in parallel. Returns None for failures.
#[pyfunction]
#[pyo3(signature = (smiles_list, n_jobs=None))]
fn encode_batch(smiles_list: Vec<String>, n_jobs: Option<usize>) -> Vec<Option<String>> {
    if let Some(n) = n_jobs {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        pool.install(|| {
            smiles_list
                .par_iter()
                .map(|s| encode::encode(s).ok())
                .collect()
        })
    } else {
        smiles_list
            .par_iter()
            .map(|s| encode::encode(s).ok())
            .collect()
    }
}

#[pymodule]
fn safe_oxidizer(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(safe_encode, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_class::<tokenizer::SafeTokenizer>()?;
    Ok(())
}
