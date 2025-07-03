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

    replay_losses = []
    validation_losses = []
    for run_id in run_ids:
        run = neptune.init_run(
            project="elise-neurotma/ELiSe", with_id=run_id, mode="read-only"
        )
        replay_loss = run["replay_loss_r"].fetch_values()["value"].tolist()
        validation_loss = run["validation_loss_r"].fetch_values()["value"].tolist()

        replay_losses.append(replay_loss)
        validation_losses.append(validation_loss)

    replay_losses = np.array(replay_losses)
    validation_losses = np.array(validation_losses)

    return replay_losses, validation_losses, run_ids


if __name__ == "__main__":
    from pathlib import Path

    tag = "silence_400_longer"
    replay_losses, validation_losses, run_ids = get_neptune_losses(tag)

    save_loc = Path(
        "/Users/benano/Documents/org/manuscripts/SequenceLearningPaper/data/silences/"
    )
    save_loc_replay = save_loc / "replay_losses.npy"
    save_loc_validation = save_loc / "validation_losses.npy"
    save_loc_config = save_loc / "config.toml"

    name = run_ids[0]

    c_run = neptune.init_run(
        project="elise-neurotma/ELiSe", with_id=name, mode="read-only"
    )
    c_run["parameters/config"].download(destination=str(save_loc_config))

    c_run["dataloader"].download(destination=str(save_loc / "dataloader.pkl"))
    c_run["network"].download(destination=str(save_loc / "network.pkl"))
    c_run["train_tracker"].download(destination=str(save_loc / "train_tracker.pkl"))
    c_run["replay_tracker"].download(destination=str(save_loc / "replay_tracker.pkl"))

    np.save(save_loc_replay, replay_losses)
    np.save(save_loc_validation, validation_losses)
