import sqlite3

from partvision import config, hf_space


def test_restore_data_copies_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTVISION_HOME", str(tmp_path / "home"))
    (config.models_dir() / "v001").mkdir(parents=True)

    def fake_download(repo_id, repo_type, local_dir):
        root = tmp_path / "snap"
        (root / "data" / "uploads").mkdir(parents=True)
        (root / "data" / "uploads" / "a.jpg").write_bytes(b"jpg")
        with sqlite3.connect(root / "data" / hf_space.DB_SNAPSHOT) as c:
            c.execute("CREATE TABLE t (x)")
        for v in ("v001", "v999"):                      # v999 is not in the model repo any more
            (root / "models" / v).mkdir(parents=True)
            (root / "models" / v / "index_extra.npz").write_bytes(b"npz")
        return str(root)

    monkeypatch.setattr(hf_space, "snapshot_download", fake_download)
    hf_space.restore_data("user/data")

    assert (config.upload_dir() / "a.jpg").read_bytes() == b"jpg"
    with sqlite3.connect(config.db_path()) as c:
        assert c.execute("SELECT name FROM sqlite_master").fetchone() == ("t",)
    assert (config.models_dir() / "v001" / "index_extra.npz").exists()
    assert not (config.models_dir() / "v999").exists()
