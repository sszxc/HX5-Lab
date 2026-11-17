from robotics.paths import asset_path, output_path, repo_root, ur5_ip


def test_paths_work_outside_checkout_and_accept_env_overrides(tmp_path, monkeypatch):
    root = repo_root()
    monkeypatch.chdir(tmp_path)
    for name in ("HX5LAB_OUTPUT_ROOT", "HX5LAB_UR5_IP"):
        monkeypatch.delenv(name, raising=False)
    assert asset_path("mujoco", "hx5_right.xml").is_file()
    assert output_path("recording.npz") == root / "out/recording.npz"
    monkeypatch.setenv("HX5LAB_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("HX5LAB_UR5_IP", "127.0.0.1")
    assert output_path() == tmp_path / "outputs"
    assert ur5_ip() == "127.0.0.1"
