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

    full_save_run = filtered_runs[
        runs_table["sys/tags"].apply(lambda tags: "full_save" in tags)
    ]
    run_save_id = full_save_run["sys/id"].tolist()[0]

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

    return replay_losses, validation_losses, run_ids, run_save_id


if __name__ == "__main__":
    from pathlib import Path

    for i in np.arange(1):
        tag_name = "final_mult_test"
        fname = "mult_pats"
        save_loc = Path(
            f"/Users/benano/Documents/org/manuscripts/SequenceLearningPaper/data/{fname}"
        )
        save_loc.mkdir(parents=True, exist_ok=True)
        # save_loc = Path("/Users/benano/Documents/testing_cluster")
        save_loc_replay = save_loc / "replay_losses.npy"
        save_loc_validation = save_loc / "validation_losses.npy"
        save_loc_config = save_loc / "config.toml"

        replay_losses, validation_losses, run_ids, run_save_id = get_neptune_losses(
            tag_name
        )
        c_run = neptune.init_run(
            project="elise-neurotma/ELiSe", with_id=run_save_id, mode="read-only"
        )
        c_run["parameters/config"].download(destination=str(save_loc_config))

        np.save(save_loc_replay, replay_losses)
        np.save(save_loc_validation, validation_losses)

        c_run["network"].download(destination=str(save_loc / "network.pkl"))
        c_run["dataloader"].download(destination=str(save_loc / "dataloader.pkl"))
        c_run["train_tracker"].download(destination=str(save_loc / "train_tracker.pkl"))
        c_run["replay_tracker"].download(
            destination=str(save_loc / "replay_tracker.pkl")
        )
        c_run["train_dict"].download(destination=str(save_loc / "train_dict.pkl"))
        c_run["validation_dict"].download(
            destination=str(save_loc / "validation_dict.pkl")
        )
        c_run["replay_dict"].download(destination=str(save_loc / "replay_dict.pkl"))
