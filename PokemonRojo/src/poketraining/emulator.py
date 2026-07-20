"""A small PyBoy adapter for Pokémon Red observations and actions."""

from __future__ import annotations

import hashlib
import io
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .config import TrainerConfig
from .memory import POKEDEX_OWNED_LENGTH, MemorySnapshot, WramAddress

UInt8Array = NDArray[np.uint8]
_CLEAN_RAM_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    """Result of applying one discrete action to the emulator."""

    observation: UInt8Array
    snapshot: MemorySnapshot
    terminated: bool


class PokemonRedEnvironment:
    """Headless Pokémon Red environment backed by PyBoy.

    The ROM must be supplied explicitly and is never copied. Calling ``close``
    uses ``save=False`` so training cannot create or overwrite cartridge RAM.
    """

    _BASE_ACTIONS = (
        ("up", WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        ("down", WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        ("left", WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        ("right", WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        ("a", WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        ("b", WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    )
    _START_ACTION = (
        "start",
        WindowEvent.PRESS_BUTTON_START,
        WindowEvent.RELEASE_BUTTON_START,
    )

    def __init__(self, rom_path: Path, config: TrainerConfig) -> None:
        self.rom_path = rom_path.expanduser().resolve()
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._pyboy: PyBoy | None = None
        self._started = False
        self._rom_fingerprints: dict[str, str] = {}
        self._clean_ram: io.BytesIO | None = None
        self._clean_rtc: io.BytesIO | None = None
        self._frame_buffer: deque[UInt8Array] = deque(maxlen=config.frame_stack)
        actions = list(self._BASE_ACTIONS)
        if config.include_start_action:
            actions.append(self._START_ACTION)
        self._actions = tuple(actions)

    @property
    def num_actions(self) -> int:
        """Return the number of actions exposed to the DQN."""

        return len(self._actions)

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return stable human-readable action labels."""

        return tuple(action[0] for action in self._actions)

    @property
    def rom_fingerprints(self) -> dict[str, str]:
        """Return hashes computed during ROM validation, without exposing its path."""

        return dict(self._rom_fingerprints)

    @property
    def pyboy(self) -> PyBoy:
        """Return the active emulator or fail with a clear lifecycle error."""

        if self._pyboy is None:
            raise RuntimeError("Environment has not been started")
        return self._pyboy

    def start(self) -> None:
        """Validate the ROM, boot the game and save a reusable playable state."""

        if self._started:
            raise RuntimeError("Environment is already started")
        self.validate_rom()
        try:
            # PyBoy otherwise auto-loads <ROM>.ram and <ROM>.rtc sidecars. Use
            # fresh in-memory cartridge storage so runs cannot consume private
            # or stale save data by accident. The maximum supported cartridge
            # RAM size is tiny compared with replay storage.
            self._clean_ram = io.BytesIO(bytes(_CLEAN_RAM_BYTES))
            self._clean_rtc = io.BytesIO()
            self._pyboy = PyBoy(
                self.rom_path.as_posix(),
                ram_file=self._clean_ram,
                rtc_file=self._clean_rtc,
                window="null",
            )
            self.pyboy.set_emulation_speed(self.config.emulation_speed)
            self._validate_cartridge_title()
            self._reach_initial_playable_state()
            # The generic wrapper does not navigate Pokémon's menus; it only
            # snapshots the current emulator state for reset_game().
            self.pyboy.game_wrapper.start_game(timer_div=self._next_timer_div())
            self._started = True
        except BaseException:
            self.close()
            raise

    def reset(self) -> tuple[UInt8Array, MemorySnapshot]:
        """Restore the saved playable state and return an initial observation."""

        if not self._started:
            raise RuntimeError("start() must be called before reset()")
        self.pyboy.game_wrapper.reset_game(timer_div=self._next_timer_div())
        random_ticks = int(self._rng.integers(0, self.config.initial_random_ticks_max + 1))
        if random_ticks:
            self._tick(random_ticks, render=True)
        else:
            self._tick(1, render=True)
        first_frame = self._processed_frame()
        self._frame_buffer = deque(
            (first_frame.copy() for _ in range(self.config.frame_stack)),
            maxlen=self.config.frame_stack,
        )
        return self._stacked_observation(), self.snapshot()

    def step(self, action_index: int) -> EnvironmentStep:
        """Apply one action and return the resulting state and memory snapshot."""

        if not 0 <= action_index < self.num_actions:
            raise ValueError(f"action_index must be in [0, {self.num_actions})")
        _, press, release = self._actions[action_index]
        self.pyboy.send_input(press)
        running = self._tick(self.config.press_ticks, render=False)
        self.pyboy.send_input(release)
        running = self._tick(self.config.release_ticks, render=True) and running
        self._frame_buffer.append(self._processed_frame())
        snapshot = self.snapshot()
        return EnvironmentStep(
            observation=self._stacked_observation(),
            snapshot=snapshot,
            terminated=(not running) or snapshot.lost_battle,
        )

    def snapshot(self) -> MemorySnapshot:
        """Read only the verified WRAM fields used by reward strategies."""

        memory = self.pyboy.memory
        start = int(WramAddress.POKEDEX_OWNED_START)
        pokedex_owned = bytes(int(memory[start + offset]) for offset in range(POKEDEX_OWNED_LENGTH))
        return MemorySnapshot(
            map_id=int(memory[int(WramAddress.CURRENT_MAP)]),
            block_y=int(memory[int(WramAddress.BLOCK_Y)]),
            block_x=int(memory[int(WramAddress.BLOCK_X)]),
            tileset_id=int(memory[int(WramAddress.TILESET)]),
            battle_type=int(memory[int(WramAddress.IS_IN_BATTLE)]),
            party_count=int(memory[int(WramAddress.PARTY_COUNT)]),
            pokedex_owned=pokedex_owned,
            enemy_level=int(memory[int(WramAddress.ENEMY_LEVEL)]),
        )

    def save_screenshot(self, path: Path) -> None:
        """Save the current RGBA screen to a caller-controlled output path."""

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(self.pyboy.screen.ndarray)).save(path)

    def close(self) -> None:
        """Stop PyBoy without persisting cartridge RAM."""

        try:
            if self._pyboy is not None:
                self._pyboy.stop(save=False)
        finally:
            self._pyboy = None
            self._started = False
            self._frame_buffer.clear()
            self._clean_ram = None
            self._clean_rtc = None

    def __enter__(self) -> PokemonRedEnvironment:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _processed_frame(self) -> UInt8Array:
        """Convert PyBoy RGB/RGBA output to an 84x84-style grayscale frame."""

        raw = np.asarray(self.pyboy.screen.ndarray, dtype=np.uint8)
        if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[-1] in (3, 4)):
            gray = Image.fromarray(raw).convert("L")
        else:
            raise ValueError(f"Unsupported PyBoy screen shape: {raw.shape}")
        height, width = self.config.frame_shape
        # BOX is an area-averaging downsampler, matching the intent of the
        # notebook's OpenCV INTER_AREA preprocessing without loading a second
        # SDL2 runtime alongside PyBoy on macOS.
        resized = gray.resize((width, height), resample=Image.Resampling.BOX)
        return np.asarray(resized, dtype=np.uint8)

    def _stacked_observation(self) -> UInt8Array:
        if len(self._frame_buffer) != self.config.frame_stack:
            raise RuntimeError("Frame buffer is not initialized")
        return np.stack(tuple(self._frame_buffer), axis=-1).astype(np.uint8, copy=False)

    def _reach_initial_playable_state(self) -> None:
        """Replay the notebook's menu sequence before saving the reset state."""

        for _ in range(self.config.skip_intro_start_presses):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
            self._tick(1, render=False)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self._tick(1, render=False)
        for _ in range(self.config.skip_intro_a_presses):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self._tick(1, render=False)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self._tick(1, render=False)
        self._tick(1, render=True)

    def _tick(self, count: int, *, render: bool) -> bool:
        if count <= 0:
            return True
        return bool(self.pyboy.tick(count, render=render))

    def _next_timer_div(self) -> int:
        return int(self._rng.integers(0, 256))

    def validate_rom(self) -> dict[str, str]:
        """Validate the local ROM and return its SHA-1/SHA-256 fingerprints."""

        self._validate_rom_file()
        return self.rom_fingerprints

    def _validate_rom_file(self) -> None:
        if not self.rom_path.is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom_path}")
        sha1 = hashlib.sha1()
        sha256 = hashlib.sha256()
        with self.rom_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha1.update(chunk)
                sha256.update(chunk)
        self._rom_fingerprints = {
            "sha1": sha1.hexdigest(),
            "sha256": sha256.hexdigest(),
        }
        if (
            self.config.expected_rom_sha1 is not None
            and self._rom_fingerprints["sha1"] != self.config.expected_rom_sha1.lower()
        ):
            raise ValueError("ROM SHA-1 does not match expected_rom_sha1")
        if (
            self.config.expected_rom_sha256 is not None
            and self._rom_fingerprints["sha256"] != self.config.expected_rom_sha256.lower()
        ):
            raise ValueError("ROM SHA-256 does not match expected_rom_sha256")

    def _validate_cartridge_title(self) -> None:
        expected = self.config.expected_cartridge_title
        if expected is None:
            return
        actual = str(self.pyboy.cartridge_title).rstrip("\x00")
        if actual != expected:
            raise ValueError(f"Unsupported cartridge title {actual!r}; expected {expected!r}")
