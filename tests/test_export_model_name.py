"""metadata.json's model_name names the model, not where the checkpoint was read from."""

from export.cli import artifact_model_name, build_parser


def test_explicit_name_wins(tmp_path):
    assert artifact_model_name(str(tmp_path), "moonshotai/Kimi-K3") == "moonshotai/Kimi-K3"
    assert artifact_model_name("other/Model", "moonshotai/Kimi-K3") == "moonshotai/Kimi-K3"


def test_hub_id_is_kept(capsys):
    assert artifact_model_name("moonshotai/Kimi-K3") == "moonshotai/Kimi-K3"
    assert capsys.readouterr().err == ""


def test_hub_cache_path(tmp_path, capsys):
    snap = tmp_path / "hub" / "models--moonshotai--Kimi-K3" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert artifact_model_name(str(snap)) == "moonshotai/Kimi-K3"
    assert artifact_model_name(str(snap) + "/") == "moonshotai/Kimi-K3"
    assert artifact_model_name(str(snap.parent.parent)) == "moonshotai/Kimi-K3"
    assert capsys.readouterr().err == ""


def test_path_falls_back_to_last_component_with_hint(tmp_path, capsys):
    ckpt = tmp_path / "k3meta"
    ckpt.mkdir()
    assert artifact_model_name(str(ckpt) + "/") == "k3meta"
    assert "--model-name" in capsys.readouterr().err


def test_remote_location_is_not_a_hub_id(capsys):
    assert artifact_model_name("gs://bucket/ckpt/Kimi-K3") == "Kimi-K3"
    assert "--model-name" in capsys.readouterr().err


def test_two_paths_of_one_model_agree(tmp_path):
    a = tmp_path / "a" / "models--moonshotai--Kimi-K3" / "snapshots" / "r1"
    b = tmp_path / "b" / "models--moonshotai--Kimi-K3" / "snapshots" / "r2"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    assert artifact_model_name(str(a)) == artifact_model_name(str(b))


def test_parser_has_model_name():
    args = build_parser().parse_args(
        ["kimi", "--output-dir", "o", "--checkpoint", "c", "--num-blocks", "1",
         "--model-name", "moonshotai/Kimi-K3"])
    assert args.model_name == "moonshotai/Kimi-K3"
    assert build_parser().parse_args(
        ["kimi", "--output-dir", "o", "--checkpoint", "c", "--num-blocks", "1"]).model_name is None
