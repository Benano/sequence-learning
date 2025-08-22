#!/usr/bin/env python3
import pickle
import tomllib as toml
from pathlib import Path

import neptune
import numpy as np


def get_neptune_losses(project_name, tag, artifact_names):
    project = neptune.init_project(project=f"elise-neurotma/{project_name}")
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
                project=f"elise-neurotma/{project_name}",
                with_id=run_id,
                mode="read-only",
            )
            run_value = run[f"{artifact_name}"].fetch_values()["value"].tolist()
            artifact_data[artifact_name].append(run_value)

        artifact_data[artifact_name] = np.array(artifact_data[artifact_name])

    return artifact_data, run_ids, run_save_id


def get_single_run(project_name, run_id, path):
    c_run = neptune.init_run(
        project=f"elise-neurotma/{project_name}", with_id=run_id, mode="read-only"
    )
    c_run["network"].download(destination=str(path / "network.pkl"))
    c_run["network"].download(destination=str(path / "network.pkl"))
    c_run["dataloader"].download(destination=str(path / "dataloader.pkl"))
    save_loc_config = path / "config.toml"
    c_run["parameters/config"].download(destination=str(save_loc_config))
    # Write a text file with the run_id to path
    with open(path / "run_id.txt", "w") as f:
        f.write(run_id)

    return c_run


def get_multi_run_group_tag(project_name, tag_names, artifact_names, save_loc):
    for tag_name in tag_names:
        c_save_loc = save_loc / tag_name
        c_save_loc.mkdir(parents=True, exist_ok=True)
        # save_loc = Path("/Users/benano/Documents/testing_cluster")
        save_loc_artifacts = c_save_loc / "artifacts.pkl"
        save_loc_config = c_save_loc / "config.toml"

        artifact_data, run_ids, run_save_id = get_neptune_losses(
            project_name, tag_name, artifact_names
        )

        c_run = neptune.init_run(
            project=f"elise-neurotma/{project_name}",
            with_id=run_save_id,
            mode="read-only",
        )
        c_run["parameters/config"].download(destination=str(save_loc_config))

        # dump artifact data to save_loc
        with open(save_loc_artifacts, "wb") as f:
            pickle.dump(artifact_data, f)

        # Save text file with the tag name
        with open(c_save_loc / "tag_name.txt", "w") as f:
            f.write(tag_name)

        c_run["network"].download(destination=str(c_save_loc / "network.pkl"))
        c_run["dataloader"].download(destination=str(c_save_loc / "dataloader.pkl"))
        c_run["train_dict"].download(destination=str(c_save_loc / "train_dict.pkl"))
        c_run["validation_dict"].download(
            destination=str(c_save_loc / "validation_dict.pkl")
        )
        c_run["replay_dict"].download(destination=str(c_save_loc / "replay_dict.pkl"))
        # c_run["epoch_dict"].download(destination=str(save_loc / "epoch_dict.pkl"))


def get_neptune_losses_scan(
    project_name, tag, param1_name, param1_values, param2_name, param2_values
):
    project = neptune.init_project(project=f"elise-neurotma/{project_name}")
    runs_table = project.fetch_runs_table().to_pandas()
    filtered_runs = runs_table[
        runs_table["sys/group_tags"].apply(lambda tags: tag in tags)
    ]

    run_ids = filtered_runs["sys/id"].tolist()

    loss_scan = np.zeros((len(param1_values), len(param2_values)))

    for idx1, param1_value in enumerate(param1_values):
        for idx2, param2_value in enumerate(param2_values):
            # Filter runs based on the parameter values

            tag_name = f"{param1_value}_{param2_value}"
            seeded_runs = filtered_runs[
                runs_table["sys/tags"].apply(lambda tags: tag_name in tags)
            ]

            loss = []
            loss_idx = 10
            for run_id in seeded_runs["sys/id"].tolist():
                run = neptune.init_run(
                    project=f"elise-neurotma/{project_name}",
                    with_id=run_id,
                    mode="read-only",
                )
                try:
                    replay_loss = run["replay_loss_r"].fetch_values()["value"].tolist()
                except KeyError:
                    print(f"Replay loss not found for run {run_id}, skipping.")
                    continue
                loss.append(replay_loss[loss_idx])

            loss = np.array(loss)
            mean_loss = np.nanmean(loss)
            loss_scan[idx1, idx2] = mean_loss

    import matplotlib.pyplot as plt

    plt.imshow(loss_scan, cmap="viridis", aspect="auto")
    plt.colorbar(label="Mean Replay Loss")
    plt.xticks(ticks=np.arange(len(param2_values)), labels=param2_values)
    plt.yticks(ticks=np.arange(len(param1_values)), labels=param1_values)
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.title(f"Mean Replay Loss for {tag}")
    plt.tight_layout()
    plt.show()

    seeded_runs = filtered_runs[
        runs_table["sys/tags"].apply(lambda tags: "full_save" in tags)
    ]
    run_save_id = full_save_run["sys/id"].tolist()[0]

    return artifact_data, run_ids, run_save_id


if __name__ == "__main__":
    # Get the losses for a scan of runs with different parameters
    # param1_name = "pattern_duration"
    # param1_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    # param1_name = "num_vis"
    # param1_values = [40, 60, 80, 100, 120, 140, 160, 180]

    # param2_name = "num_lat"
    # param2_values = [40, 60, 80, 100, 120, 140, 160, 180]

    # seeds = [1, 2, 3, 4, 5]  # run these seeds serially for each param combo

    # # project_name = "Elise-scans-width"
    # project_name = "Elise-scans-width"
    # tag = "random_scan_width"

    # get_neptune_losses_scan(
    #     project_name=project_name,
    #     tag=tag,
    #     param1_name=param1_name,
    #     param1_values=param1_values,
    #     param2_name=param2_name,
    #     param2_values=param2_values,
    # )

    artifact_names = [
        "replay_loss_r",
        # "replay_loss_pat_0",
        # "replay_loss_pat_1",
        "validation_loss_r",
        # "validation_loss_pat_0",
        # "validation_loss_pat_1",
    ]

    tag_names = [f"noise_{s}_100" for s in [2, 4, 8]]

    get_multi_run_group_tag(
        project_name="Elise-noise",
        tag_names=tag_names,
        artifact_names=artifact_names,
        save_loc=Path(
            "/Users/benano/Documents/org/manuscripts/SequenceLearningPaper/data/noise/"
        ),
    )

    # # Get the losses for a specific run group tag
    # tag_name = "reignition_short_gap"
    # save_loc = Path(f"/Users/benano/Documents/org/manuscripts/SequenceLearningPaper/data/reignition_gap/")
