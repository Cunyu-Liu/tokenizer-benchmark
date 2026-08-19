"""Unit tests for release-27 source_database composition audit (contract 3.1)."""
from data.audit_release27 import (
    parse_db_mapping_line,
    length_bin,
    composition_counts,
    load_db_mappings,
)


def test_parse_db_mapping_line_valid():
    line = "URS000001098C\tMIRTRONDB\tMIRTRONDB:29\t6239\tpre_miRNA\tY59E1B.2\n"
    acc, db, rtype = parse_db_mapping_line(line)
    assert acc == "URS000001098C"
    assert db == "MIRTRONDB"
    assert rtype == "pre_miRNA"


def test_parse_db_mapping_line_malformed():
    assert parse_db_mapping_line("") is None
    assert parse_db_mapping_line("not-a-urs\tDB\n") is None
    assert parse_db_mapping_line("URS0000000001\n") is None  # missing db col


def test_parse_db_mapping_line_missing_rna_type():
    line = "URS000001D890\tMIRTRONDB\tMIRTRONDB:138\t7237\t\n"
    acc, db, rtype = parse_db_mapping_line(line)
    assert (acc, db) == ("URS000001D890", "MIRTRONDB")
    assert rtype == "unknown"


def test_length_bin():
    assert length_bin(10) == "<16"
    assert length_bin(16) == "16-4096"
    assert length_bin(4096) == "16-4096"
    assert length_bin(4097) == "4097-16384"
    assert length_bin(16384) == "4097-16384"
    assert length_bin(20000) == ">16384"


def test_composition_counts():
    rows = [
        {"source_database": "MIRTRONDB", "rna_type": "pre_miRNA", "length": 100},
        {"source_database": "MIRTRONDB", "rna_type": "pre_miRNA", "length": 200},
        {"source_database": "CIRCPEDIA", "rna_type": "snoRNA", "length": 5000},
    ]
    out = composition_counts(rows)
    cells = out["cells"]
    assert cells["MIRTRONDB|pre_miRNA|16-4096"] == 2
    assert cells["CIRCPEDIA|snoRNA|4097-16384"] == 1


def test_load_db_mappings(tmp_path):
    (tmp_path / "mirtrondb.tsv").write_text(
        "URS000001098C\tMIRTRONDB\tMIRTRONDB:29\t6239\tpre_miRNA\tY59E1B.2\n")
    (tmp_path / "circpedia.tsv").write_text(
        "URS0000007D24\tCIRCPEDIA\tCIRCHSA_RNY4_1\t9606\tY_RNA\tENSG00000252316.1\n")
    mapping = load_db_mappings(tmp_path)
    assert mapping["URS000001098C"] == ("MIRTRONDB", "pre_miRNA")
    assert mapping["URS0000007D24"] == ("CIRCPEDIA", "Y_RNA")
    assert len(mapping) == 2