"""Unified DQN trainer for both PokeTraining reward strategies."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import TrainerConfig
from .emulator import PokemonRedEnvironment
from .model import DQN, create_synchronized_models
from .replay import ReplayBuffer, TransitionBatch
from .rewards import RewardCalculator


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Serializable summary of one train or evaluation episode."""

    episode: int
    steps: int
    total_reward: float
    epsilon: float
    terminated: bool
    truncated: bool
    mean_loss: float | None
    reward_components: dict[str, float]
    action_counts: dict[str, int]


class PokemonAITrainer:
    """Train one DQN implementation with a configurable reward strategy."""

    def __init__(
        self,
        environment: PokemonRedEnvironment,
        config: TrainerConfig,
        run_dir: Path,
    ) -> None:
        self.environment = environment
        self.config = config
        self.run_dir = run_dir.resolve()
        self.screenshot_dir = self.run_dir / "screenshots"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._seed_everything(config.seed)
        action_seed = config.seed
        replay_seed = None if config.seed is None else config.seed + 1
        self._action_rng = np.random.default_rng(action_seed)
        self.online_model, self.target_model = create_synchronized_models(
            environment.num_actions,
            config.observation_shape,
        )
        self.optimizer = keras.optimizers.Adam(config.learning_rate)
        self.loss_function = keras.losses.Huber()
        self._replay_seed = replay_seed
        # Evaluation does not need a replay buffer. Defer the historical
        # >1 GiB allocation until train() is actually requested.
        self.replay: ReplayBuffer | None = None
        self.reward_calculator = RewardCalculator(config.reward_mode)
        self.epsilon = config.epsilon_start
        self.optimizer_updates = 0
        self.environment_steps = 0

    @classmethod
    def from_rom(
        cls,
        rom_path: Path,
        config: TrainerConfig,
        run_dir: Path,
    ) -> PokemonAITrainer:
        """Construct a trainer directly from a ROM path and configuration."""

        environment = PokemonRedEnvironment(rom_path, config)
        # Fail cheaply on a missing or incompatible ROM before allocating the
        # replay buffer, which exceeds one GiB with the historical capacity.
        environment.validate_rom()
        return cls(environment, config, run_dir)

    @property
    def model(self) -> DQN:
        """Expose the online model using the notebook's familiar name."""

        return self.online_model

    def train(self) -> list[EpisodeMetrics]:
        """Run configured training episodes and always close PyBoy safely."""

        self._prepare_run_directory()
        self.replay = ReplayBuffer(
            self.config.replay_capacity,
            self.config.observation_shape,
            rng=np.random.default_rng(self._replay_seed),
        )
        metrics: list[EpisodeMetrics] = []
        try:
            self.environment.start()
            for episode_index in range(self.config.episodes):
                episode = self._train_episode(episode_index)
                metrics.append(episode)
                self._append_metrics(self.metrics_path, episode)
                print(
                    f"Episode {episode.episode} - Steps: {episode.steps} - "
                    f"Reward: {episode.total_reward:.2f}, "
                    f"Epsilon: {episode.epsilon:.3f}"
                )
                completed = episode_index + 1
                if (
                    self.config.checkpoint_every > 0
                    and completed % self.config.checkpoint_every == 0
                ):
                    self._save_checkpoint(completed)

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.online_model.save_weights(self.checkpoint_dir / "dqn_final.weights.h5")
            self._save_training_state()
            from .reporting import save_training_figures

            save_training_figures(
                metrics,
                self.environment.action_names,
                self.run_dir / "figures",
            )
        finally:
            self.environment.close()
        return metrics

    def play(
        self,
        weights_path: Path,
        *,
        episodes: int = 5,
        max_steps: int = 500,
    ) -> list[EpisodeMetrics]:
        """Evaluate saved weights greedily, resetting between episodes."""

        if episodes <= 0 or max_steps <= 0:
            raise ValueError("episodes and max_steps must be greater than zero")
        weights_path = weights_path.expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        weights_sha256 = self._sha256_file(weights_path)
        self._prepare_run_directory(input_artifacts={"weights_sha256": weights_sha256})
        self.online_model.load_weights(weights_path)
        play_metrics_path = self.run_dir / "play_metrics.jsonl"
        results: list[EpisodeMetrics] = []
        try:
            self.environment.start()
            for episode_index in range(episodes):
                state, snapshot = self.environment.reset()
                self.reward_calculator.reset(snapshot)
                total_reward = 0.0
                components: defaultdict[str, float] = defaultdict(float)
                action_counts: defaultdict[str, int] = defaultdict(int)
                terminated = False
                truncated = False
                steps = 0

                for step_index in range(max_steps):
                    action = self.choose_action(state, greedy=True)
                    action_counts[self.environment.action_names[action]] += 1
                    transition = self.environment.step(action)
                    breakdown = self.reward_calculator.calculate(
                        state[..., -1],
                        transition.observation[..., -1],
                        transition.snapshot,
                    )
                    total_reward += breakdown.total
                    for name, value in breakdown.components.items():
                        components[name] += value
                    steps = step_index + 1
                    terminated = transition.terminated
                    truncated = steps >= max_steps and not terminated
                    state = transition.observation

                    if (
                        self.config.screenshot_interval > 0
                        and step_index % self.config.screenshot_interval == 0
                    ):
                        self.environment.save_screenshot(
                            self.screenshot_dir
                            / f"play_ep{episode_index:03d}_step{step_index:05d}.png"
                        )
                    if terminated or truncated:
                        break

                episode = EpisodeMetrics(
                    episode=episode_index,
                    steps=steps,
                    total_reward=total_reward,
                    epsilon=0.0,
                    terminated=terminated,
                    truncated=truncated,
                    mean_loss=None,
                    reward_components=dict(components),
                    action_counts=dict(action_counts),
                )
                results.append(episode)
                self._append_metrics(play_metrics_path, episode)
        finally:
            self.environment.close()
        return results

    def choose_action(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        """Select an epsilon-greedy action without mutating epsilon."""

        if not greedy and self._action_rng.random() < self.epsilon:
            return int(self._action_rng.integers(self.environment.num_actions))
        q_values = self.online_model(
            np.expand_dims(observation, axis=0),
            training=False,
        )[0]
        return int(tf.argmax(q_values).numpy())

    def synchronize_target(self) -> None:
        """Copy online weights into the target network."""

        self.target_model.set_weights(self.online_model.get_weights())

    def _train_episode(self, episode_index: int) -> EpisodeMetrics:
        if self.replay is None:
            raise RuntimeError("Replay buffer is not initialized; call train()")
        state, snapshot = self.environment.reset()
        self.reward_calculator.reset(snapshot)
        total_reward = 0.0
        episode_losses: list[float] = []
        components: defaultdict[str, float] = defaultdict(float)
        action_counts: defaultdict[str, int] = defaultdict(int)
        terminated = False
        truncated = False
        steps = 0

        for step_index in range(self.config.max_steps):
            action = self.choose_action(state)
            action_counts[self.environment.action_names[action]] += 1
            transition = self.environment.step(action)
            steps = step_index + 1
            terminated = transition.terminated
            truncated = steps >= self.config.max_steps and not terminated
            breakdown = self.reward_calculator.calculate(
                state[..., -1],
                transition.observation[..., -1],
                transition.snapshot,
            )
            total_reward += breakdown.total
            for name, value in breakdown.components.items():
                components[name] += value

            self.replay.store(
                state,
                action,
                breakdown.total,
                transition.observation,
                terminated=terminated,
                truncated=truncated,
            )
            state = transition.observation
            self.environment_steps += 1

            # Complete transitions can be sampled as soon as one full batch is
            # available; the legacy +1 threshold only served its adjacency hack.
            if len(self.replay) >= self.config.batch_size:
                batch = self.replay.sample(self.config.batch_size)
                loss = self._optimize(batch)
                episode_losses.append(loss)
                self.optimizer_updates += 1
                if self.optimizer_updates % self.config.update_target_every == 0:
                    self.synchronize_target()

            # Epsilon decays globally and is deliberately not reset per episode.
            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon * self.config.epsilon_decay,
            )

            if (
                self.config.screenshot_interval > 0
                and step_index % self.config.screenshot_interval == 0
            ):
                self.environment.save_screenshot(
                    self.screenshot_dir / f"ep{episode_index:03d}_step{step_index:05d}.png"
                )
            if terminated or truncated:
                break

        return EpisodeMetrics(
            episode=episode_index,
            steps=steps,
            total_reward=total_reward,
            epsilon=self.epsilon,
            terminated=terminated,
            truncated=truncated,
            mean_loss=(float(np.mean(episode_losses)) if episode_losses else None),
            reward_components=dict(components),
            action_counts=dict(action_counts),
        )

    def _optimize(self, batch: TransitionBatch) -> float:
        loss = self._train_step(
            tf.convert_to_tensor(batch.observations),
            tf.convert_to_tensor(batch.actions),
            tf.convert_to_tensor(batch.rewards),
            tf.convert_to_tensor(batch.next_observations),
            tf.convert_to_tensor(batch.terminated),
            tf.convert_to_tensor(batch.truncated),
        )
        return float(loss.numpy())

    @tf.function(reduce_retracing=True)
    def _train_step(
        self,
        observations: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_observations: tf.Tensor,
        terminated: tf.Tensor,
        truncated: tf.Tensor,
    ) -> tf.Tensor:
        """Apply one Double-DQN update, bootstrapping across time limits."""

        next_online_q = self.online_model(next_observations, training=False)
        next_actions = tf.argmax(next_online_q, axis=1, output_type=tf.int32)
        next_target_q = self.target_model(next_observations, training=False)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        next_indices = tf.stack((batch_indices, next_actions), axis=1)
        next_values = tf.gather_nd(next_target_q, next_indices)
        # A time limit is not an environmental terminal state. Keep its next
        # value in the Bellman target while still recording truncation in replay.
        del truncated
        dones = terminated
        targets = tf.where(
            dones,
            rewards,
            rewards + self.config.gamma * next_values,
        )

        with tf.GradientTape() as tape:
            q_values = self.online_model(observations, training=True)
            action_indices = tf.stack((batch_indices, actions), axis=1)
            selected_q = tf.gather_nd(q_values, action_indices)
            loss = self.loss_function(targets, selected_q)

        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.online_model.trainable_variables,
                strict=True,
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_pairs)
        return loss

    def _prepare_run_directory(
        self,
        *,
        input_artifacts: dict[str, str] | None = None,
    ) -> None:
        # A run is immutable by default: never append to or overwrite an
        # existing experiment directory.
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "actions": list(self.environment.action_names),
            "rom_fingerprints": getattr(self.environment, "rom_fingerprints", {}),
            "runtime": self._runtime_metadata(),
            "source_revision": self._source_revision(),
        }
        if input_artifacts:
            metadata["input_artifacts"] = input_artifacts
        (self.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _save_checkpoint(self, completed_episodes: int) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.online_model.save_weights(
            self.checkpoint_dir / f"dqn_episode_{completed_episodes:04d}.weights.h5"
        )
        self._save_training_state()

    def _save_training_state(self) -> None:
        state = {
            "epsilon": self.epsilon,
            "environment_steps": self.environment_steps,
            "optimizer_updates": self.optimizer_updates,
        }
        (self.checkpoint_dir / "training_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _append_metrics(path: Path, metrics: EpisodeMetrics) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")

    @staticmethod
    def _seed_everything(seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        tf.random.set_seed(seed)

    @staticmethod
    def _runtime_metadata() -> dict[str, Any]:
        """Collect reproducibility metadata without recording private paths."""

        dependencies: dict[str, str] = {}
        for distribution in (
            "matplotlib",
            "numpy",
            "pillow",
            "pyboy",
            "tensorflow",
        ):
            try:
                dependencies[distribution] = version(distribution)
            except PackageNotFoundError:
                dependencies[distribution] = "not-installed"
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": dependencies,
        }

    @staticmethod
    def _source_revision() -> str | None:
        """Return the local Git revision when running from a working tree."""

        source_root = Path(__file__).resolve().parents[2]
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=source_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        revision = result.stdout.strip()
        return revision if result.returncode == 0 and revision else None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Hash an input artifact without recording its local path."""

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
