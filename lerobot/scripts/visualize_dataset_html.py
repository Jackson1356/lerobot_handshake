#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
python -m lerobot.scripts.visualize_dataset_html \
    --root=./data/folder_path/handshake_dataset

local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python -m lerobot.scripts.visualize_dataset_html \
    --root=./data/folder_path/handshake_dataset \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import logging
import re
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
from flask import Flask, redirect, render_template, request, url_for

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def run_server(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def homepage(dataset=dataset):
        if dataset:
            # Use dataset name from repo_id
            dataset_name = dataset.repo_id.split("/")[-1]
            return redirect(
                url_for(
                    "show_episode",
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param = None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])

        if dataset_param:
            return redirect(
                url_for(
                    "show_episode",
                    dataset_name=dataset_param,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=[],
            lerobot_datasets=[],
        )

    @app.route("/<string:dataset_name>")
    def show_first_episode(dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_name, episode_id, dataset=dataset, episodes=episodes):
        if dataset is None:
            return "No dataset loaded. Please provide a valid dataset path.", 400

        dataset_version = str(dataset.meta._version)
        match = re.search(r"v(\d+)\.", dataset_version)
        if match:
            major_version = int(match.group(1))
            if major_version < 2:
                return "Make sure to convert your LeRobotDataset to v2 & above."

        episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
        dataset_info = {
            "repo_id": dataset.repo_id,
            "num_samples": dataset.num_frames,
            "num_episodes": dataset.num_episodes,
            "fps": dataset.fps,
        }
        
        video_paths = [
            dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
        ]
        videos_info = [
            {
                "url": url_for("static", filename=str(video_path).replace("\\", "/")),
                "filename": video_path.parent.name,
            }
            for video_path in video_paths
        ]
        tasks = dataset.meta.episodes[episode_id]["tasks"]

        videos_info[0]["language_instruction"] = tasks

        if episodes is None:
            episodes = list(range(dataset.num_episodes))

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            episode_data_csv_str=episode_data_csv_str,
            columns=columns,
            ignored_columns=ignored_columns,
        )

    app.run(host=host, port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def get_episode_data(dataset: LeRobotDataset, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    selected_columns.remove("timestamp")

    ignored_columns = []
    for column_name in selected_columns:
        shape = dataset.features[column_name]["shape"]
        shape_dim = len(shape)
        if shape_dim > 1:
            selected_columns.remove(column_name)
            ignored_columns.append(column_name)

    # init header of csv with state and action names
    header = ["timestamp"]

    for column_name in selected_columns:
        dim_state = dataset.meta.shapes[column_name][0]

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    from_idx = dataset.episode_data_index["from"][episode_index]
    to_idx = dataset.episode_data_index["to"][episode_index]
    data = (
        dataset.hf_dataset.select(range(from_idx, to_idx))
        .select_columns(selected_columns)
        .with_format("pandas")
    )

    rows = np.hstack(
        (
            np.expand_dims(data["timestamp"], axis=1),
            *[np.vstack(data[col]) for col in selected_columns[1:]],
        )
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.features:
        return None

    # get first frame index
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    # Create a temporary directory that will be automatically cleaned up
    output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )
    else:
        # Create a symlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        ln_videos_dir = static_dir / "videos"
        if not ln_videos_dir.exists():
            ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Local dataset folder path (e.g. `./data/your_username/handshake_dataset`).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    dataset = LeRobotDataset(repo_id="local_dataset", root=root, tolerance_s=tolerance_s)

    visualize_dataset_html(dataset, **vars(args))


if __name__ == "__main__":
    main()
