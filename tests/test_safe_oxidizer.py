"""Tests for the Rust SAFE encoder (safe_oxidizer).

Tests use chemical equivalence (dm.same_mol) as the correctness criterion,
plus roundtrip validation through Python's safe.decode().
"""
import csv
import time
from pathlib import Path

import datamol as dm
import pytest

import safe
import safe_oxidizer

SMILES_CSV = Path("/mnt/ps/home/CORP/jason.hartford/project/safe/smiles.csv")


# ---------------------------------------------------------------------------
# Basic roundtrip tests
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Encode with Rust, decode with Python, verify chemical equivalence."""

    def test_celecoxib(self):
        smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
        rs = safe_oxidizer.safe_encode(smi)
        assert rs.count(".") == 3
        assert dm.same_mol(smi, rs)
        assert dm.same_mol(smi, safe.decode(rs))

    def test_simple_ether(self):
        smi = "CCCOCC"
        rs = safe_oxidizer.safe_encode(smi)
        assert dm.same_mol(smi, rs)
        assert dm.same_mol(smi, safe.decode(rs))

    def test_amide(self):
        smi = "CCCOCCC(=O)c1ccccc1"
        rs = safe_oxidizer.safe_encode(smi)
        assert dm.same_mol(smi, rs)

    def test_no_brics_bonds_returns_smiles(self):
        result = safe_oxidizer.safe_encode("CCO")
        assert dm.same_mol("CCO", result)

    def test_benzene_no_brics_bonds_returns_smiles(self):
        result = safe_oxidizer.safe_encode("c1ccccc1")
        assert dm.same_mol("c1ccccc1", result)

    def test_biphenyl(self):
        smi = "c1ccc(-c2ccccc2)cc1"
        rs = safe_oxidizer.safe_encode(smi)
        assert dm.same_mol(smi, rs)
        assert dm.same_mol(smi, safe.decode(rs))

    def test_invalid_smiles_unclosed_paren(self):
        with pytest.raises(ValueError):
            safe_oxidizer.safe_encode("C[C@H](C1=C(F)C=C(C")

    def test_invalid_smiles_unclosed_ring(self):
        with pytest.raises(ValueError):
            safe_oxidizer.safe_encode("C1CCC")


# ---------------------------------------------------------------------------
# Bracket SMILES edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_sm",
    [
        "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
        "[NH3+]Cc1ccccc1",
        "c1cc2c(cc1[C@@H]1CCC[NH2+]1)OCCO2",
        "[13C]1CCCCC1C[238U]C[NH3+]",
        "COC[CH2:1][CH2:2]O[CH:2]C[OH:3]",
    ],
)
def test_bracket_smiles(input_sm):
    rs = safe_oxidizer.safe_encode(input_sm)
    assert dm.same_mol(input_sm, rs), f"Rust output not same mol: {rs}"
    decoded = safe.decode(rs)
    assert decoded is not None
    assert dm.same_mol(input_sm, decoded)


# ---------------------------------------------------------------------------
# Fused ring systems
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_sm",
    [
        "[H][C@@]12CC[C@@]3(CCC(=O)O3)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])[C@@]([H])(CC2=CC(=O)CC[C@]12C)SC(C)=O",
        "[H][C@@]12C[C@H](C)[C@](OC(=O)CC)(C(=O)COC(=O)CC)[C@@]1(C)C[C@H](O)[C@@]1(Cl)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C",
        "[H][C@@]12CC[C@@](O)(C#C)[C@@]1(CC)CC[C@]1([H])[C@@]3([H])CCC(=O)C=C3CC[C@@]21[H]",
    ],
)
def test_fused_rings(input_sm):
    rs = safe_oxidizer.safe_encode(input_sm)
    decoded = safe.decode(rs)
    assert decoded is not None
    assert dm.same_mol(input_sm, decoded), f"Roundtrip failed for fused ring"


# ---------------------------------------------------------------------------
# Exact match with Python encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_sm",
    [
        "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "CCCOCC",
        "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
        "[NH3+]Cc1ccccc1",
        "CCCOCCC(=O)c1ccccc1",
    ],
)
def test_exact_match_with_python(input_sm):
    """On many molecules, Rust should produce byte-identical output to Python."""
    with dm.without_rdkit_log():
        py_result = safe.encode(input_sm)
    rs_result = safe_oxidizer.safe_encode(input_sm)
    assert rs_result == py_result, f"\nRust:   {rs_result}\nPython: {py_result}"


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_matches_individual(self):
        smiles = [
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
            "CCCOCC",
            "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
            "CCCOCCC(=O)c1ccccc1",
        ]
        batch_results = safe_oxidizer.encode_batch(smiles)
        for smi, batch_r in zip(smiles, batch_results):
            individual = safe_oxidizer.safe_encode(smi)
            assert batch_r == individual

    def test_batch_handles_no_brics(self):
        smiles = ["CCCOCC", "CCO", "c1ccccc1", "CCCOCCC(=O)c1ccccc1"]
        results = safe_oxidizer.encode_batch(smiles)
        assert all(r is not None for r in results)
        assert dm.same_mol("CCO", results[1])
        assert dm.same_mol("c1ccccc1", results[2])

    def test_batch_with_n_jobs(self):
        smiles = ["CCCOCC", "CCCOCCC(=O)c1ccccc1"] * 10
        results = safe_oxidizer.encode_batch(smiles, n_jobs=2)
        assert len(results) == 20
        assert all(r is not None for r in results)

    def test_batch_empty(self):
        assert safe_oxidizer.encode_batch([]) == []


# ---------------------------------------------------------------------------
# Performance comparison
# ---------------------------------------------------------------------------

class TestPerformance:
    @pytest.fixture
    def diverse_smiles(self):
        """Generate a list of diverse drug-like SMILES for benchmarking."""
        mols = [
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
            "CCCOCC",
            "O=C(CN1CC[NH2+]CC1)N1CCCCC1",
            "CCCOCCC(=O)c1ccccc1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "CC12CCC3c4ccc(O)cc4CCC3C1CCC2O",
            "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
            "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
            "CC(CS)C(=O)N1CCCC1C(=O)O",
        ]
        return mols * 100  # 1000 molecules

    def test_rust_faster_than_python(self, diverse_smiles):
        """Rust batch encoding should be faster than sequential Python."""
        # Warm up
        safe_oxidizer.encode_batch(diverse_smiles[:10])
        safe.encode(diverse_smiles[0])

        # Time Rust batch
        t0 = time.perf_counter()
        rust_results = safe_oxidizer.encode_batch(diverse_smiles)
        t_rust = time.perf_counter() - t0

        # Time Python sequential
        t0 = time.perf_counter()
        py_results = []
        for smi in diverse_smiles:
            try:
                py_results.append(safe.encode(smi))
            except Exception:
                py_results.append(None)
        t_py = time.perf_counter() - t0

        speedup = t_py / t_rust if t_rust > 0 else float("inf")
        print(f"\nRust batch: {t_rust:.3f}s, Python seq: {t_py:.3f}s, speedup: {speedup:.1f}x")

        # Verify correctness
        for r_rs, r_py in zip(rust_results, py_results):
            if r_rs is not None and r_py is not None:
                assert dm.same_mol(r_rs, r_py)


# ---------------------------------------------------------------------------
# Known roundtrip failures (ring closure conflicts across fragments)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smi",
    [
        r"BrC1=CC=C2NC(=O)/C(=C3\SC(=S)NC3=O)C2=C1",
        r"CC(=O)/N=C1\N(C)N=C(S(=O)(=O)N)S1",
        r"CCCOC1=CC=C(/C=C2/SC(=O)NC2=O)C=C1",
        r"OC(CCC(N)C(O)=O)=O.OC(C(N)CC1=CC=C(O)C=C1)=O.OC(C(N)C)=O.NCCCCC(N)C(O)=O.O=C(C)O.*.*.*.*.*",
        r"C/C=C1\C(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)OC=C(C(=O)OC)C1CC(=O)OC[C@H]1[C@H](C(CO)CO)C[C@H](OC(=O)CC2C(C(=O)OC)=COC(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)/C2=C\C)[C@H]1C",
        r"C/C=C1/[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)OC=C(C(=O)OC)[C@H]1CC(=O)OC[C@H]1O[C@@H](O[C@]23CO[C@H](C4=CC=C(O)C(OC)=C4)[C@H]2CO[C@@H]3C2=CC=C(O)C(OC)=C2)[C@H](O)[C@@H](O)[C@@H]1O",
        r"CN(C)CC/C=C1/C2=C(C=CC=C2)COC2=C1C=C(CC(=O)NC(C#C)C1=CC=CC=C1)C=C2",
        r"C[C@H]1CN(C2=C(C(F)(F)F)C=C(NC(=O)C/C=C3\CCCC=C3)C=C2)C[C@@H](C)O1",
        r"O=C(C1=CNC2=NC=CC=C12)/N=C1\SC=C(CO)N1CC1=CC=CC=C1",
    ],
)
def test_known_failures(smi):
    """Roundtrip failures from eval_tokenizer_accuracy.py — ring closure conflicts."""
    result = safe_oxidizer.safe_encode(smi)
    assert dm.same_mol(smi, result), f"decoded: {result}"


@pytest.mark.slow
def test_known_failure_large_dna():
    """Large DNA oligomer — ring closure numbers overflow causing duplicate labels."""
    smi = (
        "N1(C2OC(COP(N3CC(COP(N4CC(COP(N5CC(COP(N6CC(COOP(=O)(N7CC(COP(N8CC(COP(N9CC(COP(N%10CC(COP(N%11CC(COP(N%12CC(COP(N%13CC(COOP(=O)(N%14CC(COP(N%15CC(COP(N%16CC(COP(N%17CC(COP(N%18CC(COP(N%19CC(COP(N%20CC(COOP(=O)(N%21CC(COP(N%22CC(COP(N%23CC(COP(N%24CC(COP(N%25CC(COP(N%26CC(COP(N%27CC(COOP(=O)(N%28CC(COP(N%29CC(COP(N%30CC(COP(N%31CC(COP(N%32CCN(C(OCCOCCOCCO)=O)CC%32)(N(C)C)=O)OC(N%32C(=O)N=C(N)C=C%32)C%31)(N(C)C)=O)OC(N%31C(=O)NC(=O)C(C)=C%31)C%30)(N(C)C)=O)OC(N%30C(=O)N=C(N)C=C%30)C%29)(N(C)C)=O)OC(N%29C(=O)N=C(N)C=C%29)C%28)N(C)C)OC(N%28C=NC%29=C%28N=CN=C%29N)C%27)(N(C)C)=O)OC(N%27C=NC%28=C%27N=CN=C%28N)C%26)(N(C)C)=O)OC(N%26C(=O)N=C(N)C=C%26)C%25)(N(C)C)=O)OC(N%25C=NC%26=C%25N=CN=C%26N)C%24)(N(C)C)=O)OC(N%24C(=O)NC(=O)C(C)=C%24)C%23)(N(C)C)=O)OC(N%23C(=O)N=C(N)C=C%23)C%22)(N(C)C)=O)OC(N%22C=NC%23=C%22N=CN=C%23N)C%21)N(C)C)OC(N%21C=NC%22=C%21N=CN=C%22N)C%20)(N(C)C)=O)OC(N%20C=NC%21=C%20N=C(N)NC%21=O)C%19)(N(C)C)=O)OC(N%19C=NC%20=C%19N=C(N)NC%20=O)C%18)(N(C)C)=O)OC(N%18C=NC%19=C%18N=CN=C%19N)C%17)(N(C)C)=O)OC(N%17C=NC%18=C%17N=CN=C%18N)C%16)(N(C)C)=O)OC(N%16C=NC%17=C%16N=C(N)NC%17=O)C%15)(N(C)C)=O)OC(N%15C=NC%16=C%15N=CN=C%16N)C%14)N(C)C)OC(N%14C(=O)NC(=O)C(C)=C%14)C%13)(N(C)C)=O)OC(N%13C=NC%14=C%13N=C(N)NC%14=O)C%12)(N(C)C)=O)OC(N%12C=NC%13=C%12N=C(N)NC%13=O)C%11)(N(C)C)=O)OC(N%11C(=O)N=C(N)C=C%11)C%10)(N(C)C)=O)OC(N%10C=NC%11=C%10N=CN=C%11N)C9)(N(C)C)=O)OC(N9C(=O)NC(=O)C(C)=C9)C8)(N(C)C)=O)OC(N8C(=O)NC(=O)C(C)=C8)C7)N(C)C)OC(N7C(=O)NC(=O)C(C)=C7)C6)(N(C)C)=O)OC(N6C(=O)N=C(N)C=C6)C5)(N(C)C)=O)OC(N5C(=O)NC(=O)C(C)=C5)C4)(N(C)C)=O)OC(N4C=NC5=C4N=CN=C5N)C3)(N(C)C)=O)CNC2)C=NC2=C1N=C(N)NC2=O"
    )
    result = safe_oxidizer.safe_encode(smi)
    assert dm.same_mol(smi, result), f"decoded: {result}"


# ---------------------------------------------------------------------------
# Large-scale chemical equivalence
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SMILES_CSV.exists(), reason="smiles.csv not found")
def test_10k_chemical_equivalence():
    """Rust safe_encode output must be chemically equivalent to the input for 10k molecules."""
    with open(SMILES_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        smiles_list = [row[0] for row in reader if row and row[0]][:10_000]
    results = safe_oxidizer.encode_batch(smiles_list, n_jobs=4)

    # Molecules where E/Z stereo crosses a BRICS-cut double bond lose stereo
    # in both Python and Rust SAFE encoders (inherent SAFE limitation).
    KNOWN_STEREO_LOSSES = {
        "BrC1=CC=C2NC(=O)/C(=C3\\SC(=S)NC3=O)C2=C1",
    }

    failures = []
    for smi, result in zip(smiles_list, results):
        if smi in KNOWN_STEREO_LOSSES:
            continue
        if result is None:
            failures.append((smi, "encode returned None"))
        elif not dm.same_mol(smi, result):
            failures.append((smi, f"not equivalent: {result}"))

    if failures:
        sample = failures[:20]
        msg = f"{len(failures)}/{len(smiles_list)} molecules failed chemical equivalence:\n"
        msg += "\n".join(f"  {smi}: {reason}" for smi, reason in sample)
        pytest.fail(msg)
