use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;

use ahash::{AHashMap, AHashSet};
use numpy::PyArray2;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::encode;

type Pair = (u32, u32);

const NUM_SPECIAL_TOKENS: u32 = 5;

#[derive(Serialize, Deserialize)]
struct TokenizerData {
    version: u32,
    merges: Vec<((u32, u32), u32)>,
    num_merges: u32,
    #[serde(default)]
    additional_tokens: Vec<String>,
}

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

/// Split a SMILES into BRICS fragments via SAFE encoding.
/// Falls back to treating the entire SMILES as one fragment on failure.
fn pre_tokenize(smiles: &str) -> Vec<Vec<u8>> {
    match encode::encode(smiles) {
        Ok(safe_str) => safe_str
            .split('.')
            .map(|s| s.as_bytes().to_vec())
            .collect(),
        Err(_) => vec![smiles.as_bytes().to_vec()],
    }
}

#[pyclass]
pub struct SafeTokenizer {
    pub merges: HashMap<Pair, u32>,
    pub num_merges: u32,
    pub additional_tokens: Vec<String>,
    // not serialized; rebuilt on load and after mutation
    additional_token_lookup: HashMap<String, u32>,
}

impl SafeTokenizer {
    fn rebuild_lookup(&mut self) {
        let base = 256 + self.num_merges + NUM_SPECIAL_TOKENS;
        self.additional_token_lookup = self
            .additional_tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), base + i as u32))
            .collect();
    }

    fn train_core_incremental(
        &mut self,
        mut words: Vec<Word>,
        counts: Vec<i32>,
        vocab_size: u32,
    ) {
        assert!(
            vocab_size >= 256 + NUM_SPECIAL_TOKENS,
            "vocab_size must be at least {}",
            256 + NUM_SPECIAL_TOKENS
        );
        let num_merges = vocab_size - 256 - NUM_SPECIAL_TOKENS;
        self.merges.clear();

        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        let mut heap = BinaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        let mut merges_done = 0u32;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break };

            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if current <= 0 {
                continue;
            }
            if top.count != current as u64 {
                top.count = current as u64;
                heap.push(top);
                continue;
            }

            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;
        }

        self.num_merges = merges_done;
    }

    fn build_vocab(&self) -> Vec<Vec<u8>> {
        let total = 256 + self.num_merges as usize + NUM_SPECIAL_TOKENS as usize
            + self.additional_tokens.len();
        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
        vocab.resize(total, Vec::new());

        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&(left, right), &merged_id) in &sorted_merges {
            let mut merged_bytes = vocab[left as usize].clone();
            merged_bytes.extend(&vocab[right as usize]);
            vocab[merged_id as usize] = merged_bytes;
        }

        let base = 256 + self.num_merges;
        vocab[base as usize] = b".".to_vec();
        vocab[(base + 1) as usize] = b"<pad>".to_vec();
        vocab[(base + 2) as usize] = b"<bos>".to_vec();
        vocab[(base + 3) as usize] = b"<eos>".to_vec();
        vocab[(base + 4) as usize] = b"<unk>".to_vec();

        for (i, s) in self.additional_tokens.iter().enumerate() {
            vocab[(base + 5 + i as u32) as usize] = s.as_bytes().to_vec();
        }

        vocab
    }

    fn encode_fragment(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        while ids.len() >= 2 {
            let mut best: Option<(Pair, u32)> = None;
            for i in 0..ids.len() - 1 {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if best.is_none() || new_id < best.unwrap().1 {
                        best = Some((pair, new_id));
                    }
                }
            }

            let Some((pair, new_id)) = best else { break };

            let (a, b) = pair;
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
        }

        ids
    }

    fn encode_smiles(&self, smiles: &str) -> Vec<u32> {
        if let Some(&id) = self.additional_token_lookup.get(smiles) {
            return vec![id];
        }
        let fragments = pre_tokenize(smiles);
        self.encode_fragments(&fragments)
    }

    fn encode_smiles_shuffled(&self, smiles: &str) -> Vec<u32> {
        if let Some(&id) = self.additional_token_lookup.get(smiles) {
            return vec![id];
        }
        let mut fragments = pre_tokenize(smiles);
        fragments.shuffle(&mut thread_rng());
        self.encode_fragments(&fragments)
    }

    fn encode_fragments(&self, fragments: &[Vec<u8>]) -> Vec<u32> {
        let dot_id = self.dot_token_id_inner();
        let mut result = Vec::new();

        for (i, frag) in fragments.iter().enumerate() {
            if i > 0 {
                result.push(dot_id);
            }
            result.extend(self.encode_fragment(frag));
        }

        result
    }

    fn decode_ids(&self, ids: &[u32]) -> String {
        let vocab = self.build_vocab();
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = vocab.get(id as usize) {
                bytes.extend(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[inline]
    fn dot_token_id_inner(&self) -> u32 {
        256 + self.num_merges
    }

    fn vocab_size_inner(&self) -> u32 {
        256 + self.num_merges + NUM_SPECIAL_TOKENS + self.additional_tokens.len() as u32
    }
}

unsafe impl Send for SafeTokenizer {}
unsafe impl Sync for SafeTokenizer {}

#[pymethods]
impl SafeTokenizer {
    #[new]
    pub fn new() -> Self {
        Self {
            merges: HashMap::new(),
            num_merges: 0,
            additional_tokens: Vec::new(),
            additional_token_lookup: HashMap::new(),
        }
    }

    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192))]
    pub fn train_from_iterator(
        &mut self,
        py: Python<'_>,
        iterator: &Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
    ) -> PyResult<()> {
        let mut py_iter = iterator.try_iter()?;

        let mut counts: AHashMap<Vec<u8>, i32> = AHashMap::new();
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        loop {
            // Refill buffer under GIL
            buf.clear();
            let mut exhausted = false;
            for _ in 0..buffer_size {
                match py_iter.next() {
                    Some(obj) => {
                        let s: String = obj?.extract()?;
                        buf.push(s);
                    }
                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }

            if buf.is_empty() && exhausted {
                break;
            }

            // Release GIL for parallel SAFE-encode + fragment counting
            let local: AHashMap<Vec<u8>, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<Vec<u8>, i32> = AHashMap::new();
                        for frag in pre_tokenize(s) {
                            *m.entry(frag).or_default() += 1;
                        }
                        m
                    })
                    .reduce(AHashMap::new, |mut a, b| {
                        for (k, v) in b {
                            *a.entry(k).or_default() += v;
                        }
                        a
                    })
            });

            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }

        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(chunk.iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }

        self.train_core_incremental(words, cvec, vocab_size);
        Ok(())
    }

    #[pyo3(signature = (smiles, add_special_tokens=false))]
    pub fn encode(&self, smiles: &str, add_special_tokens: bool) -> Vec<u32> {
        let mut ids = self.encode_smiles(smiles);
        if add_special_tokens {
            ids.insert(0, self.bos_token_id());
            ids.push(self.eos_token_id());
        }
        ids
    }

    pub fn decode(&self, ids: Vec<u32>) -> String {
        self.decode_ids(&ids)
    }

    #[pyo3(signature = (smiles_list, max_length=None, shuffle=false, add_special_tokens=false))]
    pub fn batch_encode<'py>(
        &self,
        py: Python<'py>,
        smiles_list: Vec<String>,
        max_length: Option<usize>,
        shuffle: bool,
        add_special_tokens: bool,
    ) -> Bound<'py, PyArray2<u32>> {
        if smiles_list.is_empty() {
            return PyArray2::zeros(py, [0, 0], false);
        }

        let bos_id = self.bos_token_id();
        let eos_id = self.eos_token_id();

        let encoded: Vec<Vec<u32>> = py.allow_threads(|| {
            smiles_list
                .par_iter()
                .map(|s| {
                    let mut ids = if shuffle {
                        self.encode_smiles_shuffled(s)
                    } else {
                        self.encode_smiles(s)
                    };
                    if add_special_tokens {
                        if let Some(n) = max_length {
                            ids.truncate(n.saturating_sub(2));
                        }
                        ids.insert(0, bos_id);
                        ids.push(eos_id);
                    }
                    ids
                })
                .collect()
        });

        let pad_len = match max_length {
            Some(n) => n,
            None => encoded.iter().map(|v| v.len()).max().unwrap_or(0),
        };

        let pad_id = self.pad_token_id();

        let padded: Vec<Vec<u32>> = encoded
            .iter()
            .map(|seq| {
                let mut row = vec![pad_id; pad_len];
                let copy_len = seq.len().min(pad_len);
                row[..copy_len].copy_from_slice(&seq[..copy_len]);
                row
            })
            .collect();

        PyArray2::from_vec2(py, &padded).unwrap()
    }

    pub fn batch_decode(&self, py: Python<'_>, ids_list: Vec<Vec<u32>>) -> Vec<String> {
        let vocab = self.build_vocab();
        py.allow_threads(|| {
            ids_list
                .par_iter()
                .map(|ids| {
                    let mut bytes = Vec::new();
                    for &id in ids {
                        if let Some(token_bytes) = vocab.get(id as usize) {
                            bytes.extend(token_bytes);
                        }
                    }
                    String::from_utf8_lossy(&bytes).into_owned()
                })
                .collect()
        })
    }

    #[getter]
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size_inner()
    }

    #[getter]
    pub fn dot_token_id(&self) -> u32 {
        self.dot_token_id_inner()
    }

    #[getter]
    pub fn pad_token_id(&self) -> u32 {
        256 + self.num_merges + 1
    }

    #[getter]
    pub fn bos_token_id(&self) -> u32 {
        256 + self.num_merges + 2
    }

    #[getter]
    pub fn eos_token_id(&self) -> u32 {
        256 + self.num_merges + 3
    }

    #[getter]
    pub fn unk_token_id(&self) -> u32 {
        256 + self.num_merges + 4
    }

    pub fn add_tokens(&mut self, tokens: Vec<String>) {
        for t in tokens {
            if !self.additional_token_lookup.contains_key(&t) {
                self.additional_tokens.push(t.clone());
            }
        }
        self.rebuild_lookup();
    }

    #[getter]
    pub fn num_additional_tokens(&self) -> usize {
        self.additional_tokens.len()
    }

    pub fn get_vocab(&self) -> Vec<(u32, Vec<u8>)> {
        let vocab = self.build_vocab();
        vocab
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect()
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let mut merges: Vec<_> = self.merges.iter().map(|(&k, &v)| (k, v)).collect();
        merges.sort_by_key(|&(_, v)| v);

        let data = TokenizerData {
            version: 1,
            merges,
            num_merges: self.num_merges,
            additional_tokens: self.additional_tokens.clone(),
        };

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| PyIOError::new_err(format!("Serialization error: {}", e)))?;

        fs::write(Path::new(path), json)
            .map_err(|e| PyIOError::new_err(format!("Write error: {}", e)))?;

        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<SafeTokenizer> {
        let json = fs::read_to_string(Path::new(path))
            .map_err(|e| PyIOError::new_err(format!("Read error: {}", e)))?;

        let data: TokenizerData = serde_json::from_str(&json)
            .map_err(|e| PyIOError::new_err(format!("Deserialization error: {}", e)))?;

        let merges: HashMap<(u32, u32), u32> = data.merges.into_iter().collect();

        let mut tok = SafeTokenizer {
            merges,
            num_merges: data.num_merges,
            additional_tokens: data.additional_tokens,
            additional_token_lookup: HashMap::new(),
        };
        tok.rebuild_lookup();
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_tokenize_simple() {
        let frags = pre_tokenize("CCO");
        assert_eq!(frags.len(), 1);
    }

    #[test]
    fn test_pre_tokenize_brics() {
        let frags = pre_tokenize("CCCOCC");
        assert!(
            frags.len() >= 2,
            "Expected >=2 fragments, got {}",
            frags.len()
        );
    }

    #[test]
    fn test_pre_tokenize_invalid() {
        let frags = pre_tokenize("not_a_smiles???");
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0], b"not_a_smiles???");
    }

    #[test]
    fn test_word_merge_pair() {
        let mut word = Word::new(vec![1, 2, 3, 1, 2]);
        let _deltas = word.merge_pair((1, 2), 99);
        assert_eq!(word.ids, vec![99, 3, 99]);
    }

    #[test]
    fn test_word_merge_no_match() {
        let mut word = Word::new(vec![1, 2, 3]);
        let deltas = word.merge_pair((4, 5), 99);
        assert_eq!(word.ids, vec![1, 2, 3]);
        assert!(deltas.is_empty());
    }

    #[test]
    fn test_train_small() {
        let mut tok = SafeTokenizer::new();
        let words = vec![Word::new(vec![97, 98]), Word::new(vec![99, 100])];
        let counts = vec![10, 5];
        tok.train_core_incremental(words, counts, 262);
        assert_eq!(tok.num_merges, 1);
        assert!(tok.merges.contains_key(&(97, 98)));
        assert_eq!(*tok.merges.get(&(97, 98)).unwrap(), 256);
    }

    #[test]
    fn test_train_chained_merges() {
        let mut tok = SafeTokenizer::new();
        let words = vec![Word::new(vec![97, 97, 97])];
        let counts = vec![10];
        tok.train_core_incremental(words, counts, 263);
        assert_eq!(tok.num_merges, 2);
        assert_eq!(*tok.merges.get(&(97, 97)).unwrap(), 256);
        assert_eq!(*tok.merges.get(&(256, 97)).unwrap(), 257);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut tok = SafeTokenizer::new();
        tok.merges.insert((67, 67), 256);
        tok.num_merges = 1;
        tok.rebuild_lookup();

        let ids = tok.encode_smiles("CCO");
        let decoded = tok.decode_ids(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_dot_tokens_between_fragments() {
        let tok = SafeTokenizer {
            merges: HashMap::new(),
            num_merges: 0,
            additional_tokens: Vec::new(),
            additional_token_lookup: HashMap::new(),
        };
        let dot_id = tok.dot_token_id_inner();

        let ids = tok.encode_smiles("CCCOCC");
        let dot_count = ids.iter().filter(|&&id| id == dot_id).count();
        let frags = pre_tokenize("CCCOCC");
        assert_eq!(dot_count, frags.len() - 1);
    }

    #[test]
    fn test_special_token_ids_consistent() {
        let tok = SafeTokenizer {
            merges: HashMap::new(),
            num_merges: 10,
            additional_tokens: Vec::new(),
            additional_token_lookup: HashMap::new(),
        };
        assert_eq!(tok.dot_token_id_inner(), 266);
        assert_eq!(tok.vocab_size_inner(), 256 + 10 + 5);
    }

    #[test]
    fn test_encode_fragment_no_merges() {
        let tok = SafeTokenizer {
            merges: HashMap::new(),
            num_merges: 0,
            additional_tokens: Vec::new(),
            additional_token_lookup: HashMap::new(),
        };
        let ids = tok.encode_fragment(b"CC");
        assert_eq!(ids, vec![67, 67]);
    }

    #[test]
    fn test_encode_fragment_with_merges() {
        let mut merges = HashMap::new();
        merges.insert((67, 67), 256u32);
        let tok = SafeTokenizer {
            merges,
            num_merges: 1,
            additional_tokens: Vec::new(),
            additional_token_lookup: HashMap::new(),
        };
        let ids = tok.encode_fragment(b"CCC");
        assert_eq!(ids, vec![256, 67]);
    }
}
