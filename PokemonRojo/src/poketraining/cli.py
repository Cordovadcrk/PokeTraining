"""Command-line entry points for training and evaluating PokeTraining."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import REWARD_MODES, TrainerConfig, load_config

DEFAULT_OUTPUT_ROOT = Path("results") / "runs"
_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def build_parser() -> argparse.ArgumentParser:
    """Build the public ``train``/``play`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="poketraining",
        description="Train or evaluate a DQN agent with a user-supplied Pokémon Red ROM.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new DQN agent.")
    _add_common_arguments(train_parser)
    train_parser.add_argument("--episodes", type=int, help="Override training episodes.")
    train_parser.add_argument(
        "--max-steps",
        type=int,
        help="Override the maximum number of steps per episode.",
    )

    play_parser = subparsers.add_parser("play", help="Evaluate saved DQN weights.")
    _add_common_arguments(play_parser)
    play_parser.add_argument("--weights", type=Path, required=True)
    play_parser.add_argument("--episodes", type=int, default=5)
    play_parser.add_argument("--max-steps", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process exit status."""

    args = build_parser().parse_args(argv)
    config = _apply_overrides(load_config(args.config), args)
    if args.command == "play":
        args.weights = args.weights.expanduser().resolve()
        if not args.weights.is_file():
            raise FileNotFoundError(f"Weights not found: {args.weights}")
    # Keep ``--help`` fast and quiet; heavy native dependencies are needed only
    # once an actual train/play command has been fully parsed.
    environment_class, trainer_class = _runtime_classes()

    run_dir = _new_run_directory(
        args.output_root,
        args.run_name,
        command=args.command,
        reward_mode=config.reward_mode,
    )
    environment = environment_class(args.rom, config)
    # Validate before the trainer allocates the replay arrays and DQN weights.
    environment.validate_rom()
    trainer = trainer_class(environment, config, run_dir)

    if args.command == "train":
        trainer.train()
        print(f"Training artifacts: {run_dir}")
        return 0
    trainer.play(
        args.weights,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print(f"Evaluation artifacts: {run_dir}")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rom",
        type=Path,
        required=True,
        help="Path to a legally obtained Pokémon Red ROM; never copied to outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Run root (default: results/runs).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional unique directory name using letters, numbers, '.', '_' or '-'.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=sorted(REWARD_MODES),
        help="Override the configured reward strategy.",
    )
    parser.add_argument("--seed", type=int, help="Optional reproducibility seed.")
    parser.add_argument(
        "--buffer-size",
        type=int,
        help="Override replay capacity (useful for low-memory smoke runs).",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=int,
        help="Steps between screenshots; 0 disables them.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="Episodes between checkpoints; 0 disables intermediate checkpoints.",
    )
    parser.add_argument(
        "--include-start-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include START in the action space.",
    )


def _runtime_classes() -> tuple[type, type]:
    """Import native/heavy runtime classes only for an executable command."""

    from .emulator import PokemonRedEnvironment
    from .trainer import PokemonAITrainer

    return PokemonRedEnvironment, PokemonAITrainer


def _apply_overrides(config: TrainerConfig, args: argparse.Namespace) -> TrainerConfig:
    updates: dict[str, object] = {}
    for argument, field_name in (
        ("reward_mode", "reward_mode"),
        ("seed", "seed"),
        ("episodes", "episodes"),
        ("max_steps", "max_steps"),
        ("buffer_size", "replay_capacity"),
        ("screenshot_interval", "screenshot_interval"),
        ("checkpoint_every", "checkpoint_every"),
        ("include_start_action", "include_start_action"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates) if updates else config


def _new_run_directory(
    output_root: Path,
    run_name: str | None,
    *,
    command: str,
    reward_mode: str,
) -> Path:
    root = output_root.expanduser().resolve()
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{timestamp}_{command}_{reward_mode}"
    elif not _RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run-name may contain only letters, numbers, '.', '_' and '-'")
    run_dir = root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir
