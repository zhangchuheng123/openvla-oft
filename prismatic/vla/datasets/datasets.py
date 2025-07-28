"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import os
import json
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from numpy.lib.stride_tricks import sliding_window_view


EPS = 1e-6


from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights



@dataclass
class RobotBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""

        dataset_name =  "dataset_v6"  # rlds_batch["dataset_name"]
        current_action = rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0].numpy())
        lang = rlds_batch["task"]["language_instruction"].lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # debug RobotBatchTransform
        import pdb; pdb.set_trace()

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RobotDataset(torch.utils.data.IterableDataset):
    """
    train_dataset = RobotDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    """

    def __init__(self, config, batch_transform, train=True, num_workers=4):
        
        self.batch_transform = batch_transform

        self.proprio_type = config.proprio_type
        self.action_type = config.action_type 

        if self.proprio_type == 'joint':
            self.proprio_key = 'joint'
            self.proprio_chunk_key = 'joint_chunk'
        elif self.proprio_type == 'poseulerg':
            self.proprio_key = 'proprio'
            self.proprio_chunk_key = 'proprio_chunk'

        self.wrist_key = config.wrist_key
        self.image_key = config.image_key
        self.force_regenerate = config.force_regenerate_meta
        self.plot_hist = config.plot_hist
        self.action_chunk_size = config.action_chunk_size
        self.proprio_hist_size = config.proprio_hist_size
        self.proprio_chunk_size = config.proprio_chunk_size
        self.extension = getattr(config, 'extension', 'jpg')
        self.overwrite_stats = getattr(config, 'overwrite_stats', 'real_robot_v1')
        self.data_path = Path(config.data_path) 

        spec = f's{self.proprio_type}_{self.proprio_hist_size}_{self.proprio_chunk_size}_a{self.action_type}_{self.action_chunk_size}'

        if train:
            meta_path = os.path.join(self.data_path, f'metadata_train_{spec}_v3.pkl')
        else:
            meta_path = os.path.join(self.data_path, f'metadata_val_{spec}_v3.pkl')

        if os.path.isfile(meta_path) and (not self.force_regenerate):
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print(f'generating {meta_path}...')
            self.metadata = self._generate_metadata(train)

            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)

        stat_path = os.path.join(self.data_path, 'dataset_statistics_train.pkl')
        if os.path.isfile(stat_path) and ((not self.force_regenerate) or (not train)):
            with open(stat_path, "r", encoding="utf-8") as f:
                self.dataset_statistics = json.load(f)
        else:
            assert train, "dataset statistics can only be generated on training dataset"
            self.dataset_statistics = self._generate_statistics(self.metadata)
            with open(stat_path, "w", encoding="utf-8") as f:
                json.dump(self.dataset_statistics, f, indent=4) 

        self.dataset_statistics = self._overwrite_statistics(self.dataset_statistics)
        self.weights = [item['num_steps'] for item in self.metadata]

        if self.plot_hist:
            self.plot_hist_prop_action()

        self.img_transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), scale=[0.8, 1.0], ratio=[0.9, 1.1], antialias=True), 
            v2.ColorJitter(brightness=0.1, contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=0.05),
        ])

        print(f"#trajs={len(self.weights)} #steps={sum(self.weights)} avg steps per traj={np.mean(self.weights):.1f}")

    def plot_hist_prop_action(self):
        proprios = [item['proprio'] for item in self.metadata]
        actions = [item['action'] for item in self.metadata]

        proprios = np.concatenate(proprios)
        actions = np.concatenate(actions)
        # proprios and actions shape: [B, 7] 
        self._plot_hist_single(proprios, actions, 'original_hist.png')

        normed_proprios = self._normalize(proprios, self.dataset_statistics['proprio'])
        normed_actions = self._normalize(actions, self.dataset_statistics['action'])
        # normed_proprios and normed_proprios shape: [B, 7] 
        self._plot_hist_single(normed_proprios, normed_actions, 'normalized_hist.png')

    def _plot_hist_single(self, proprios, actions, filename):
        # create a fig with figsize=(15, 6)
        plt.figure(figsize=(15, 6))
        
        # Plot proprios dimensions (first row)
        for dim in range(7):
            plt.subplot(2, 7, dim + 1)
            plt.hist(proprios[:, dim], bins=50, color='skyblue', alpha=0.7)
            plt.title(f'Proprio Dim {dim}')
            plt.grid(True, linestyle='--', alpha=0.5)

        # Plot actions dimensions (second row)
        for dim in range(7):
            plt.subplot(2, 7, dim + 8)  # 8-14 for second row
            plt.hist(actions[:, dim], bins=50, color='salmon', alpha=0.7)
            plt.title(f'Action Dim {dim}')
            plt.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.data_path, filename)
        print(f'Saving to {output_path}')
        plt.savefig(output_path)
        plt.close()

    def _overwrite_statistics(self, stats):

        if self.overwrite_stats == 'real_robot_v1':
            # Temporary fix for current data
            # Single step move ~< 5cm; rotate ~< 5.8 deg = 0.1 rad
            # WARNING: These parameters only suited for v6 dataset
            stats['action']['q99'] = [ 0.05,  0.05,  0.05,  0.1,  0.1,  0.1, 1000]
            stats['action']['q01'] = [-0.05, -0.05, -0.05, -0.1, -0.1, -0.1, 0]

            # roll, pitch, yaw ranges from [-pi, +pi]
            stats['proprio']['q99'][3:] = [ 3.15,  3.15,  3.15, 1000]
            stats['proprio']['q01'][3:] = [-3.15, -3.15, -3.15, 0   ]

        elif self.overwrite_stats == 'touchsim':

            stats['action']['q99'][3:-1] = [ EPS,  EPS,  EPS]
            stats['action']['q01'][3:-1] = [-EPS, -EPS, -EPS]

        for key in ['action', 'proprio', 'joint', 'length']:
            for stype in ['mean', 'std', 'max', 'min', 'q01', 'q99']:
                stats[key][stype] = np.array(stats[key][stype])

        return stats 

    def _generate_statistics(self, metadata):

        joints = [item.get('joint', np.zeros((1, 7))) for item in metadata]
        proprios = [item['proprio'] for item in metadata]
        actions = [item['action'] for item in metadata]
        lengths = [item['num_steps'] for item in metadata]

        joints = np.concatenate(joints)
        proprios = np.concatenate(proprios)
        actions = np.concatenate(actions)
        lengths = np.array(lengths)

        statistics = {
            "action": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "max": actions.max(0).tolist(),
                "min": actions.min(0).tolist(),
                "q01": np.quantile(actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            },
            "proprio": {
                "mean": proprios.mean(0).tolist(),
                "std": proprios.std(0).tolist(),
                "max": proprios.max(0).tolist(),
                "min": proprios.min(0).tolist(),
                "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
                "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
            },
            "joint": {
                "mean": joints.mean(0).tolist(),
                "std": joints.std(0).tolist(),
                "max": joints.max(0).tolist(),
                "min": joints.min(0).tolist(),
                "q01": np.quantile(joints, 0.01, axis=0).tolist(),
                "q99": np.quantile(joints, 0.99, axis=0).tolist(),
            },
            "length": {
                "mean": lengths.mean()[None].tolist(),
                "std": lengths.std()[None].tolist(),
                "max": lengths.max()[None].tolist(),
                "min": lengths.min()[None].tolist(),
                "q01": np.quantile(lengths, 0.01)[None].tolist(),
                "q99": np.quantile(lengths, 0.99)[None].tolist(),
            },
            "num_transitions": len(metadata),
            "num_trajectories": int(lengths.sum()),
        }

        return statistics

    def _generate_metadata(self, train=True, train_ratio=0.9):
        
        traj_list = sorted([child for child in self.data_path.iterdir() \
            if child.is_dir() and not child.is_symlink()])
        num_trajs = len(traj_list)

        train_ratio_x10 = int(train_ratio * 10)

        if train:
            traj_list = [val for ind, val in enumerate(traj_list) if ind % 10 < train_ratio_x10]
        else:
            traj_list = [val for ind, val in enumerate(traj_list) if ind % 10 >= train_ratio_x10]

        print(f'process {len(traj_list)}/{num_trajs} trajectories')

        metadata = []
        for traj in tqdm(traj_list):

            joint_file = traj / 'left_arm_joint_status.npy'
            if joint_file.exists():
                joint = np.load(joint_file)
            else: 
                joint = None

            proprio_file = traj / 'left_arm_poseuler_arm.npy'
            if not proprio_file.exists():
                proprio_file = traj / 'proprio.npy'
            proprio = np.load(proprio_file)

            action_file = traj / 'action.npy'
            action = np.load(action_file)

            # temporary fix for not including gripper state in proprio
            if proprio.shape[1] == 6:
                proprio = np.hstack([proprio, action[:, [-1]]])
                proprio[1:, -1] = proprio[:-1, -1]

            obs_folder = traj / 'images' 
            img_folder = obs_folder / self.image_key 
            num_steps = sum(1 for _ in img_folder.glob(f"*.{self.extension}"))

            if 'panda_v2' in str(self.data_path):
                # temporary fix for touchsim_panda_v2
                if proprio.shape[0] + 1 == num_steps:
                    num_steps -= 1

            assert proprio.shape[0] == action.shape[0]
            assert proprio.shape[0] == num_steps

            image_hist_chunk = np.arange(num_steps).reshape(-1, 1)
            image_hist_chunk = np.pad(image_hist_chunk, ((self.proprio_hist_size, self.proprio_chunk_size - 1), (0, 0)), mode='edge')
            image_hist_chunk = sliding_window_view(image_hist_chunk, self.proprio_hist_size + self.proprio_chunk_size, 0)
            image_hist_chunk = rearrange(image_hist_chunk, 'B dim C -> B C dim')

            proprio_hist_chunk = np.pad(proprio, ((self.proprio_hist_size, self.proprio_chunk_size - 1), (0, 0)), mode='edge')
            proprio_hist_chunk = sliding_window_view(proprio_hist_chunk, self.proprio_hist_size + self.proprio_chunk_size, 0)
            proprio_hist_chunk = rearrange(proprio_hist_chunk, 'B dim C -> B C dim')

            if joint is not None:
                joint_hist_chunk = np.pad(joint, ((self.proprio_hist_size, self.proprio_chunk_size - 1), (0, 0)), mode='edge')
                joint_hist_chunk = sliding_window_view(joint_hist_chunk, self.proprio_hist_size + self.proprio_chunk_size, 0)
                joint_hist_chunk = rearrange(joint_hist_chunk, 'B dim C -> B C dim')

            action_chunk = np.pad(action, ((0, self.action_chunk_size - 1), (0, 0)), mode='edge')            
            action_chunk = sliding_window_view(action_chunk, self.action_chunk_size, 0)
            action_chunk = rearrange(action_chunk, 'B dim C -> B C dim')

            instruction_file = traj / 'task_instruction.txt'
            if os.path.isfile(instruction_file):
                with open(instruction_file, 'r') as f:
                    lang = f.read()
            else:
                lang = ''

            meta = {
                'proprio': proprio,
                'action': action,
                'proprio_chunk': proprio_hist_chunk,
                'image_steps': image_hist_chunk,
                'action_chunk': action_chunk,
                'num_steps': num_steps,
                'lang_instr': lang,
                'image_path': str(obs_folder),
            }

            if joint is not None:
                meta.update({
                    'joint': joint,
                    'joint_chunk': joint_hist_chunk,
                })

            metadata.append(meta)
        return metadata

    @staticmethod
    def _normalize(input_vector, stats, method='q01q99'):

        vector = input_vector.copy()

        if method == 'minmax':
            vector = (vector - stats['min']) / (stats['max'] - stats['min'])
            vector = (vector - 0.5) * 2
        elif method == 'q01q99':
            vector = (vector - stats['q01']) / (stats['q99'] - stats['q01'])
            vector = (vector - 0.5) * 2
        elif method == 'meanstd':
            vector = (vector - stats['mean']) / (stats['std'] + EPS)

        return vector

    def normalize(self, input_vector, key='action', method='q01q99'):
        if key == 'action':
            norm_key = 'action'
        elif key == 'proprio':
            norm_key = self.proprio_key
        
        return self._normalize(input_vector, self.dataset_statistics[norm_key], method)

    @staticmethod
    def _denormalize(input_vector, stats, method='q01q99'):

        vector = input_vector.copy()

        if method == 'minmax':
            vector = vector / 2 + 0.5
            vector = vector * (stats['max'] - stats['min']) + stats['min']
        elif method == 'q01q99':
            vector = vector / 2 + 0.5
            vector = vector * (stats['q99'] - stats['q01']) + stats['q01']
        elif method == 'meanstd':
            vector = vector * (stats['std'] + EPS) + stats['mean']

        return vector

    def denormalize(self, input_vector, key='action', method='q01q99'):
        if key == 'action':
            norm_key = 'action'
        elif key == 'proprio':
            norm_key = self.proprio_key
        
        return self._denormalize(input_vector, self.dataset_statistics[norm_key], method)

    def _get_images(self, image_folder, step_list, image_key):
        images = []
        for i in step_list:
            image_path = Path(image_folder) / image_key / f"{i}.{self.extension}"
            images.append(v2.PILToTensor()(Image.open(image_path)))

        images = torch.stack(images)
        images = self.img_transform(images)
        images = rearrange(images, 'chunk C H W -> chunk H W C')
        return images

    def _get_data(self, sample, index):

        image_path = sample['image_path']
        image_steps = sample['image_steps'][index].flatten().tolist()

        image_primary = self._get_images(image_path, image_steps, self.image_key)

        sample_dict = {
            'observation': {
                'image_primary': image_primary,
                'proprio': self._normalize(sample[self.proprio_chunk_key][index], self.dataset_statistics[self.proprio_key]),
            },
            'action': self._normalize(sample['action_chunk'][index], self.dataset_statistics['action']),
            'task':{
                'language_instruction': sample['lang_instr'],
            },
        }
        if self.wrist_key is not None:
            image_wrist = self._get_images(image_path, image_steps, self.wrist_key)
            sample_dict['observation']['image_wrist'] = image_wrist

        return sample_dict

    def __iter__(self):

        for _ in range(len(self.metadata)):

            sample = random.choices(self.metadata, weights=self.weights)[0]
            index = random.randint(0, sample['num_steps'] - 1)

            yield self.batch_transform(self._get_data(sample, index))

    def __len__(self):
        return len(self.metadata)


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
