"""Seed plumbing through the fine-tuning CLI and Slurm, without real GPU jobs."""

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tyro

from scripts import gr00t_finetune as training

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", [None, 43])
def test_seed_cli_defaults_and_override(seed):
    args = ["--dataset-path", "unused"]
    if seed is not None:
        args += ["--seed", str(seed)]
    config = tyro.cli(training.ArgsConfig, args=args)
    assert config.seed == (42 if seed is None else seed)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_training_seed_rejected_before_loading_data(seed):
    with pytest.raises(ValueError, match="seed"):
        training.ArgsConfig(dataset_path=["unused"], seed=seed)


def test_seed_reaches_initialization_mixture_and_trainer(tmp_path, monkeypatch):
    from gr00t.model.transforms import GR00TTransform

    # Skip the image processor: this test exercises main's orchestration, not VLM loading.
    last = GR00TTransform.model_construct(eagle_processor=None, max_action_dim=32)
    transforms = SimpleNamespace(transforms=[last])
    data_config = SimpleNamespace(
        action_indices=list(range(16)), modality_config=lambda: {}, transform=lambda: transforms
    )
    monkeypatch.setattr(training, "load_data_config", lambda value: data_config)
    monkeypatch.setattr(training, "LeRobotSingleDataset", lambda **kwargs: SimpleNamespace())
    original_set_seed = training.set_seed
    captured = {}

    def set_seed(seed):
        captured["initialization_seed"] = seed
        original_set_seed(seed)

    def mixture(**kwargs):
        captured["mixture_seed"] = kwargs["seed"]
        captured["mixture_sample"] = np.random.default_rng(kwargs["seed"]).integers(0, 100000, 8)
        return SimpleNamespace()

    def load_model(**kwargs):
        captured["base_model"] = kwargs["pretrained_model_name_or_path"]
        captured["initial_weights"] = torch.rand(8)
        return SimpleNamespace(
            action_head=SimpleNamespace(config=SimpleNamespace(action_horizon=16, action_dim=32)),
            config=SimpleNamespace(),
        )

    class Runner:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs["training_args"]
            captured["resume"] = kwargs["resume_from_checkpoint"]

        def train(self, save_final_model):
            captured["save_final_model"] = save_final_model

    monkeypatch.setattr(training, "set_seed", set_seed)
    monkeypatch.setattr(training, "LeRobotMixtureDataset", mixture)
    monkeypatch.setattr(training, "GR00T_N1_5", SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr(training, "TrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(training, "TrainRunner", Runner)
    paths = [tmp_path / "object", tmp_path / "goal"]
    for path in paths:
        path.mkdir()
    snapshots = []
    for seed in [42, 42, 43]:
        config = training.ArgsConfig(
            dataset_path=[str(p) for p in paths],
            seed=seed,
            base_model_path="original-pretrained-base",
            num_train_epochs=2,
            max_steps=-1,
            save_strategy="epoch",
            save_total_limit=2,
            save_final_model=False,
        )
        training.main(config)
        args = captured["training_args"]
        assert captured["initialization_seed"] == captured["mixture_seed"] == args.seed == args.data_seed == seed
        assert captured["base_model"] == "original-pretrained-base"
        assert not captured["resume"] and not captured["save_final_model"]
        assert (args.num_train_epochs, args.max_steps, args.save_strategy, args.save_total_limit) == (
            2,
            -1,
            "epoch",
            2,
        )
        snapshots.append((captured["initial_weights"].clone(), captured["mixture_sample"].copy()))
    assert torch.equal(snapshots[0][0], snapshots[1][0])
    np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
    assert not torch.equal(snapshots[0][0], snapshots[2][0])
    assert not np.array_equal(snapshots[0][1], snapshots[2][1])


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def launch_environment(tmp_path):
    data = tmp_path / "datasets"
    for suite in ["spatial", "object", "goal", "long"]:
        metadata = data / suite / "meta"
        metadata.mkdir(parents=True)
        (metadata / "modality.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = dict(os.environ)
    env.pop("GR00T_TRAIN_SEED", None)
    env.pop("GR00T_LAUNCH_MODE", None)
    env.update(
        LIBERO_DATA_ROOT=str(data),
        GR00T_PYTHON=sys.executable,
        LOSO_RUN_ROOT=str(tmp_path / "output with spaces"),
        SLURM_SUBMIT_DIR=str(REPO),
        SLURM_JOB_ID="123456",
        GR00T_BASE_MODEL=str(tmp_path / "original-base"),
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
    )
    return env, bin_dir


def test_submission_dry_run_passes_seed_without_writing_or_submitting(launch_environment):
    env, _ = launch_environment
    result = subprocess.run(
        ["bash", str(REPO / "scripts/submit_libero_loso.sh"), "--seed", "43", "--dry-run", "spatial"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("sbatch ")))
    assert command[-3:] == ["spatial", env["LOSO_RUN_ROOT"], "43"]
    assert "--job-name=gr00t-loso-spatial-s43" in command
    assert not Path(env["LOSO_RUN_ROOT"]).exists()


@pytest.mark.parametrize("seed", [None, 43])
def test_submission_records_seed_and_forwards_to_worker_using_fake_sbatch(launch_environment, seed):
    env, bin_dir = launch_environment
    capture = bin_dir.parent / "sbatch_args.json"
    env["TEST_SBATCH_CAPTURE"] = str(capture)
    executable(
        bin_dir / "sbatch",
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\nPath(os.environ['TEST_SBATCH_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('987654')\n",
    )
    args = [] if seed is None else [f"--seed={seed}"]
    result = subprocess.run(
        ["bash", str(REPO / "scripts/submit_libero_loso.sh"), *args, "spatial"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = 42 if seed is None else seed
    assert json.loads(capture.read_text())[-3:] == ["spatial", env["LOSO_RUN_ROOT"], str(expected)]
    with (Path(env["LOSO_RUN_ROOT"]) / "submissions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1 and rows[0]["seed"] == str(expected) and rows[0]["job_id"] == "987654"


@pytest.mark.parametrize("seed", ["-1", "4294967296", "abc"])
def test_submission_rejects_invalid_seeds_without_outputs(launch_environment, seed):
    env, _ = launch_environment
    result = subprocess.run(
        ["bash", str(REPO / "scripts/submit_libero_loso.sh"), "--seed", seed, "--dry-run", "spatial"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "seed must be" in result.stderr
    assert not Path(env["LOSO_RUN_ROOT"]).exists()


@pytest.mark.parametrize("seed", [None, 43])
def test_slurm_worker_keeps_seed_in_manifest_training_and_wandb(launch_environment, seed):
    env, bin_dir = launch_environment
    capture = bin_dir.parent / "training_args.json"
    env["TEST_TRAIN_CAPTURE"] = str(capture)
    # Replace only the Slurm launch. No actual GR00T training or W&B connection occurs.
    executable(
        bin_dir / "srun",
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["TEST_TRAIN_CAPTURE"]).write_text(json.dumps(args))
output = Path(args[args.index("--output-dir") + 1])
for epoch in [1, 2]:
    checkpoint = output / f"checkpoint-{epoch * 10}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"epoch": epoch, "global_step": epoch * 10}))
""",
    )
    executable(bin_dir / "nvidia-smi", "#!/bin/sh\necho 'Fake GPU, 81920 MiB'\n")
    args = [] if seed is None else [str(seed)]
    command = ["bash", str(REPO / "scripts/libero_loso.slurm"), "spatial", env["LOSO_RUN_ROOT"], *args]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = 42 if seed is None else seed
    output = Path(env["LOSO_RUN_ROOT"]) / "heldout_spatial"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["seed"] == manifest["data_seed"] == expected
    assert manifest["execution_mode"] == "slurm"
    assert f"-s{expected}-" in manifest["wandb_run_id"]
    assert Path(manifest["base_model"]) == Path(env["GR00T_BASE_MODEL"])
    assert {Path(p).name for p in manifest["training_datasets"]} == {"object", "goal", "long"}
    training_args = json.loads(capture.read_text())
    assert training_args[training_args.index("--seed") + 1] == str(expected)
    assert training_args[training_args.index("--num-train-epochs") + 1] == "2"
    assert training_args[training_args.index("--save-total-limit") + 1] == "2"
    assert "--no-save-final-model" in training_args
    assert [c["epoch"] for c in json.loads((output / "checkpoints.json").read_text())] == [1, 2]
    # A second invocation must refuse the existing checkpoints before training.
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0 and "Refusing to overwrite" in repeated.stderr


def sequential_training_stubs(launch_environment):
    env, bin_dir = launch_environment
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["TEST_SEQUENCE_EVENTS"] = str(bin_dir.parent / "training_events.jsonl")
    env["TEST_CUDA_PROBE"] = str(bin_dir.parent / "cuda_probe")
    env["TEST_UNEXPECTED_SLURM"] = str(bin_dir.parent / "unexpected_slurm")
    wrapper = bin_dir / "fake_python"
    executable(
        wrapper,
        f"#!{sys.executable}\n"
        + """import json, os, sys, time
from pathlib import Path
if sys.argv[1:2] == ["-c"]:
    Path(os.environ["TEST_CUDA_PROBE"]).write_text("checked")
    if os.environ.get("TEST_FAKE_CUDA_FAIL"):
        raise SystemExit("fake CUDA unavailable")
    print("Using fake allocated GPU")
    raise SystemExit(0)
if sys.argv[1:2] == ["scripts/gr00t_finetune.py"]:
    args = sys.argv[2:]
    output = Path(args[args.index("--output-dir") + 1])
    suite = output.name.removeprefix("heldout_")
    events = Path(os.environ["TEST_SEQUENCE_EVENTS"])
    lock = events.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with events.open("a") as stream:
            stream.write(json.dumps({"event": "start", "suite": suite, "args": args,
                                     "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
        time.sleep(0.05)
        if os.environ.get("TEST_FAIL_FOLD") == suite:
            raise SystemExit(7)
        for epoch in [1, 2]:
            checkpoint = output / f"checkpoint-{epoch * 10}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(json.dumps({"epoch": epoch, "global_step": epoch * 10}))
        with events.open("a") as stream:
            stream.write(json.dumps({"event": "end", "suite": suite}) + "\\n")
        print(f"Finished fake training: {suite}")
    finally:
        os.close(descriptor)
        lock.unlink()
    raise SystemExit(0)
os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])
""".replace("REAL_PYTHON", repr(sys.executable)),
    )
    env["GR00T_PYTHON"] = str(wrapper)
    for name in ["sbatch", "srun"]:
        executable(
            bin_dir / name,
            f"#!{sys.executable}\nimport os\nfrom pathlib import Path\n"
            f"Path(os.environ['TEST_UNEXPECTED_SLURM']).write_text('{name}')\nraise SystemExit(99)\n",
        )
    executable(bin_dir / "nvidia-smi", "#!/bin/sh\necho 'Fake GPU, 81920 MiB'\n")
    return env


def test_sequential_training_uses_current_gpu_in_order_without_slurm(launch_environment):
    env = sequential_training_stubs(launch_environment)
    command = ["bash", str(REPO / "scripts/train_libero_loso_sequential.sh"), "--seed", "43"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    events = [json.loads(line) for line in Path(env["TEST_SEQUENCE_EVENTS"]).read_text().splitlines()]
    suites = ["spatial", "object", "goal", "long"]
    assert [(event["event"], event["suite"]) for event in events] == [
        (phase, suite) for suite in suites for phase in ["start", "end"]
    ]
    root = Path(env["LOSO_RUN_ROOT"])
    assert Path(env["TEST_CUDA_PROBE"]).exists()
    assert not Path(env["TEST_UNEXPECTED_SLURM"]).exists()
    for event in events[::2]:
        assert event["cuda"] == "3"
        args = event["args"]
        assert args[args.index("--seed") + 1] == "43"
        assert args[args.index("--num-train-epochs") + 1] == "2"
        assert args[args.index("--save-total-limit") + 1] == "2"
        datasets = args[args.index("--dataset-path") + 1 : args.index("--data-config")]
        assert {Path(p).name for p in datasets} == set(suites) - {event["suite"]}
        folder = root / f"heldout_{event['suite']}"
        manifest = json.loads((folder / "run_manifest.json").read_text())
        assert manifest["execution_mode"] == "direct" and manifest["seed"] == 43
        assert manifest["slurm_job_id"] == "123456"
        assert len(json.loads((folder / "checkpoints.json").read_text())) == 2
        assert (root / f"heldout_{event['suite']}.log").is_file()
    with (root / "sequence.tsv").open() as stream:
        statuses = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["status"] for row in statuses] == [status for _ in suites for status in ["running", "completed"]]
    previous = Path(env["TEST_SEQUENCE_EVENTS"]).read_text()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0 and "Refusing to overwrite" in repeated.stderr
    assert Path(env["TEST_SEQUENCE_EVENTS"]).read_text() == previous


def test_sequential_training_stops_on_failure(launch_environment):
    env = sequential_training_stubs(launch_environment)
    env["TEST_FAIL_FOLD"] = "object"
    result = subprocess.run(
        ["bash", str(REPO / "scripts/train_libero_loso_sequential.sh"), "--seed", "43"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    root = Path(env["LOSO_RUN_ROOT"])
    events = [json.loads(line) for line in Path(env["TEST_SEQUENCE_EVENTS"]).read_text().splitlines()]
    assert [(event["event"], event["suite"]) for event in events] == [
        ("start", "spatial"),
        ("end", "spatial"),
        ("start", "object"),
    ]
    with (root / "sequence.tsv").open() as stream:
        statuses = list(csv.DictReader(stream, delimiter="\t"))
    assert statuses[-1]["status"] == "failed" and statuses[-1]["exit_code"] == "7"
    assert not (root / "heldout_goal").exists() and not (root / "heldout_long").exists()
    assert not Path(env["TEST_UNEXPECTED_SLURM"]).exists()


def test_sequential_dry_run_does_not_probe_gpu_or_create_outputs(launch_environment):
    env = sequential_training_stubs(launch_environment)
    result = subprocess.run(
        ["bash", str(REPO / "scripts/train_libero_loso_sequential.sh"), "--seed", "43", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("GR00T_LAUNCH_MODE=direct bash") == 4
    assert not Path(env["TEST_CUDA_PROBE"]).exists()
    assert not Path(env["LOSO_RUN_ROOT"]).exists()
    assert not Path(env["TEST_SEQUENCE_EVENTS"]).exists()


def test_sequential_training_checks_cuda_before_starting(launch_environment):
    env = sequential_training_stubs(launch_environment)
    env["TEST_FAKE_CUDA_FAIL"] = "1"
    result = subprocess.run(
        ["bash", str(REPO / "scripts/train_libero_loso_sequential.sh"), "--seed", "43"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "CUDA unavailable" in result.stderr
    assert not Path(env["LOSO_RUN_ROOT"]).exists()
    assert not Path(env["TEST_SEQUENCE_EVENTS"]).exists()
