#!/usr/bin/env python3
import neptune
import numpy as np


def get_neptune_losses(tag):
    project = neptune.init_project(project="elise-neurotma/ELiSe")
    runs_table = project.fetch_runs_table().to_pandas()
    filtered_runs = runs_table[
        runs_table["sys/group_tags"].apply(lambda tags: tag in tags)
    ]

    run_ids = filtered_runs["sys/id"].tolist()

    results = {"replay_losses": [], "validation_losses": []}
    for run_id in run_ids:
        run = neptune.init_run(
            project="elise-neurotma/ELiSe", with_id=run_id, mode="read-only"
        )
        replay_loss = run["replay_loss_r"].fetch_values()["value"].tolist()
        validation_loss = run["validation_loss_r"].fetch_values()["value"].tolist()

        results["replay_losses"].append(replay_loss)
        results["validation_losses"].append(validation_loss)

    results["replay_losses"] = np.array(results["replay_losses"])
    results["validation_losses"] = np.array(results["validation_losses"])

    return results


if __name__ == "__main__":
    tag = "multiple_durations_150"
    losses = get_neptune_losses(tag)
