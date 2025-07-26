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


class FinetuneDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(self, config, train=True, num_workers=4):
        
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

            yield self._get_data(sample, index)

    def __len__(self):
        return len(self.metadata)