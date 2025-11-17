from src.data.ingest import build_manifest


def test_build_manifest_has_expected_columns_and_rows():
    df = build_manifest()
    # basic shape sanity
    assert not df.empty
    # critical columns for reproducibility
    for col in ["period", "relative_path", "filename", "file_type", "sha256"]:
        assert col in df.columns


