import json

import numpy as np
import pytest

from scripts.rag.analyze_context_rollouts import (
    correlation,
    describe,
    load_episodes,
    plot,
)


def write_episode(root, index, scores, complete=True):
    directory = root / "uncertainty" / f"task03_init{index:03d}"
    directory.mkdir(parents=True)
    metadata = dict(
        complete=complete,
        task_id=3,
        episode_index=index,
        success=index == 0,
        control_steps=len(scores) * 16,
        num_queries=len(scores),
        score_signature=dict(score_id="test", action_horizon=16),
    )
    (directory / "episode.json").write_text(json.dumps(metadata))
    rows = []
    for i, score in enumerate(scores):

        def entry(value):
            return dict(
                u_ctx=value,
                score_id="test",
                u_ctx_per_dimension=[value / 7] * 7,
                context=dict(task="task", indices=[0], similarities=[0.8]),
                normalized_action_distance_to_executed=1.0,
            )

        row = entry(score)
        row.update(
            control_step=i * 16,
            context_diagnostics={
                "random_same_task": entry(score + 1),
                "low_similarity_same_task": entry(score + 2),
            },
        )
        rows.append(row)
    (directory / "queries.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return directory


def test_analysis_balances_episodes_and_plots_saved_traces(tmp_path):
    write_episode(tmp_path, 0, [2.0])
    write_episode(tmp_path, 1, [4.0, 8.0])
    write_episode(tmp_path, 2, [999.0], complete=False)
    episodes = load_episodes(tmp_path)
    metrics = describe(episodes)
    assert metrics["num_episodes"] == 2 and metrics["num_queries"] == 3
    assert metrics["context_sources"]["retrieved"]["episode_balanced_mean"] == 4.0
    assert (
        metrics["context_sources"]["random_same_task"][
            "episode_balanced_mean_paired_delta"
        ]
        == 1.0
    )
    assert (
        metrics["context_sources"]["random_same_task"][
            "episode_balanced_fraction_above_retrieved"
        ]
        == 1.0
    )
    output = tmp_path / "plots"
    output.mkdir()
    plot(episodes, output)
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.pdf"))) == 4
    assert correlation(np.ones(3), np.arange(3), True) is None


def test_analysis_refuses_missing_queries_or_mixed_signatures(tmp_path):
    directory = write_episode(tmp_path, 0, [2.0, 3.0])
    path = directory / "queries.jsonl"
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="query steps"):
        load_episodes(tmp_path)
    row = json.loads(lines[1])
    row["score_id"] = "different"
    path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="signatures"):
        load_episodes(tmp_path)
