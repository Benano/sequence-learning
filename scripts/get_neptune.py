#!/usr/bin/env python3
import pickle

import neptune
import numpy as np


def get_neptune_losses(tag, artifact_names):
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

    artifact_data = {}
    for artifact_name in artifact_names:
        artifact_data[artifact_name] = []
        for run_id in run_ids:
            run = neptune.init_run(
                project="elise-neurotma/ELiSe", with_id=run_id, mode="read-only"
            )
            run_value = run[f"{artifact_name}"].fetch_values()["value"].tolist()
            artifact_data[artifact_name].append(run_value)

        artifact_data[artifact_name] = np.array(artifact_data[artifact_name])

    return artifact_data, run_ids, run_save_id


if __name__ == "__main__":
    from pathlib import Path

    for i in np.arange(1):
        tag_name = "mult_learning"
        fname = "mult_pats"
        save_loc = Path(
            f"/Users/benano/Documents/org/manuscripts/SequenceLearningPaper/data/{fname}"
        )
        save_loc.mkdir(parents=True, exist_ok=True)
        # save_loc = Path("/Users/benano/Documents/testing_cluster")
        save_loc_artifacts = save_loc / "artifacts.pkl"
        save_loc_config = save_loc / "config.toml"

        artifact_names = [
            "replay_loss_r",
            "replay_loss_pat_0",
            "replay_loss_pat_1",
            "validation_loss_r",
            "validation_loss_pat_0",
            "validation_loss_pat_1",
        ]
        artifact_data, run_ids, run_save_id = get_neptune_losses(
            tag_name, artifact_names
        )

        c_run = neptune.init_run(
            project="elise-neurotma/ELiSe", with_id=run_save_id, mode="read-only"
        )
        c_run["parameters/config"].download(destination=str(save_loc_config))

        # dump artifact data to save_loc
        with open(save_loc_artifacts, "wb") as f:
            pickle.dump(artifact_data, f)

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
