"""ROM-free adapter tests using an in-memory PyBoy-shaped fake."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyboy")

from poketraining import emulator as emulator_module
from poketraining.config import TrainerConfig
from poketraining.emulator import PokemonRedEnvironment
from poketraining.memory import POKEDEX_OWNED_LENGTH, WramAddress


class FakePyBoy:
    """Minimal object exposing only fields exercised by adapter unit tests."""

    def __init__(self, frame: np.ndarray, memory: np.ndarray | None = None) -> None:
        self.screen = SimpleNamespace(ndarray=frame)
        self.memory = memory if memory is not None else np.zeros(65_536, dtype=np.uint8)
        self.cartridge_title = "POKEMON RED\x00"
        self.stop_calls: list[bool] = []

    def stop(self, *, save: bool) -> None:
        self.stop_calls.append(save)


def environment_for_frame(frame: np.ndarray, *, height: int, width: int):
    config = TrainerConfig(
        frame_height=height,
        frame_width=width,
        frame_stack=2,
        batch_size=1,
        replay_capacity=1,
        expected_rom_sha1=None,
    )
    environment = PokemonRedEnvironment(Path("unused.gb"), config)
    environment._pyboy = FakePyBoy(frame)  # type: ignore[assignment]
    return environment


def test_rgb_and_rgba_frames_produce_same_grayscale_values() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[10, 20, 30], [50, 60, 70], [200, 100, 0]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array([[0, 64, 128], [192, 254, 255]], dtype=np.uint8)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1)
    rgb_environment = environment_for_frame(rgb, height=36, width=40)
    rgba_environment = environment_for_frame(rgba, height=36, width=40)

    rgb_gray = rgb_environment._processed_frame()
    rgba_gray = rgba_environment._processed_frame()

    assert rgb_gray.shape == (36, 40)
    assert rgb_gray.dtype == np.uint8
    np.testing.assert_array_equal(rgb_gray, rgba_gray)


def test_grayscale_input_is_resized_to_height_width_order() -> None:
    source = np.arange(8, dtype=np.uint8).reshape(2, 4)
    environment = environment_for_frame(source, height=36, width=40)

    processed = environment._processed_frame()

    assert processed.shape == (36, 40)
    assert processed.dtype == np.uint8


def test_unsupported_screen_shape_is_rejected() -> None:
    environment = environment_for_frame(
        np.zeros((2, 2, 2), dtype=np.uint8),
        height=36,
        width=36,
    )

    with pytest.raises(ValueError, match="Unsupported PyBoy screen shape"):
        environment._processed_frame()


def test_snapshot_reads_verified_addresses_and_full_pokedex_bitfield() -> None:
    memory = np.zeros(65_536, dtype=np.uint8)
    values = {
        WramAddress.CURRENT_MAP: 7,
        WramAddress.BLOCK_Y: 8,
        WramAddress.BLOCK_X: 9,
        WramAddress.TILESET: 10,
        WramAddress.IS_IN_BATTLE: 2,
        WramAddress.PARTY_COUNT: 4,
        WramAddress.ENEMY_LEVEL: 16,
    }
    for address, value in values.items():
        memory[int(address)] = value
    pokedex = bytes(range(POKEDEX_OWNED_LENGTH))
    start = int(WramAddress.POKEDEX_OWNED_START)
    memory[start : start + POKEDEX_OWNED_LENGTH] = np.frombuffer(
        pokedex,
        dtype=np.uint8,
    )
    environment = environment_for_frame(
        np.zeros((2, 2, 3), dtype=np.uint8),
        height=36,
        width=36,
    )
    environment._pyboy = FakePyBoy(  # type: ignore[assignment]
        np.zeros((2, 2, 3), dtype=np.uint8),
        memory,
    )

    snapshot = environment.snapshot()

    assert snapshot.map_id == 7
    assert snapshot.block_id == (7, 9, 8)
    assert snapshot.tileset_id == 10
    assert snapshot.in_battle is True
    assert snapshot.party_count == 4
    assert snapshot.enemy_level == 16
    assert snapshot.pokedex_owned == pokedex


def test_action_set_is_stable_and_start_is_opt_in() -> None:
    base = PokemonRedEnvironment(
        Path("unused.gb"),
        TrainerConfig(expected_rom_sha1=None),
    )
    with_start = PokemonRedEnvironment(
        Path("unused.gb"),
        TrainerConfig(include_start_action=True, expected_rom_sha1=None),
    )

    assert base.action_names == ("up", "down", "left", "right", "a", "b")
    assert with_start.action_names == (*base.action_names, "start")


def test_cartridge_title_validation_strips_null_padding() -> None:
    environment = environment_for_frame(
        np.zeros((2, 2), dtype=np.uint8),
        height=36,
        width=36,
    )

    environment._validate_cartridge_title()

    environment.pyboy.cartridge_title = "POKEMON BLUE"
    with pytest.raises(ValueError, match="Unsupported cartridge title"):
        environment._validate_cartridge_title()


def test_close_never_persists_cartridge_ram() -> None:
    environment = environment_for_frame(
        np.zeros((2, 2), dtype=np.uint8),
        height=36,
        width=36,
    )
    fake = environment.pyboy

    environment.close()

    assert fake.stop_calls == [False]
    with pytest.raises(RuntimeError, match="not been started"):
        _ = environment.pyboy


def test_start_uses_clean_in_memory_ram_before_saving_reset_state(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[object] = []
    constructor: dict[str, object] = {}

    class FakeWrapper:
        def start_game(self, *, timer_div: int) -> None:
            events.append(("saved_reset_state", timer_div))

    class BootPyBoy:
        cartridge_title = "POKEMON RED"

        def __init__(self) -> None:
            self.game_wrapper = FakeWrapper()

        def set_emulation_speed(self, speed: int) -> None:
            events.append(("speed", speed))

        def send_input(self, event: object) -> None:
            events.append(("input", event))

        def tick(self, count: int, *, render: bool) -> bool:
            events.append(("tick", count, render))
            return True

        def stop(self, *, save: bool) -> None:
            events.append(("stop", save))

    def fake_pyboy(path: str, **kwargs: object) -> BootPyBoy:
        constructor.update({"path": path, **kwargs})
        return BootPyBoy()

    monkeypatch.setattr(emulator_module, "PyBoy", fake_pyboy)
    rom_path = tmp_path / "POKEMON_RED.gb"
    rom_path.write_bytes(b"synthetic ROM bytes for lifecycle test")
    config = TrainerConfig(
        expected_rom_sha1=None,
        expected_rom_sha256=None,
        skip_intro_start_presses=1,
        skip_intro_a_presses=1,
        seed=7,
    )
    environment = PokemonRedEnvironment(rom_path, config)

    environment.start()

    ram_file = constructor["ram_file"]
    rtc_file = constructor["rtc_file"]
    assert isinstance(ram_file, emulator_module.io.BytesIO)
    assert len(ram_file.getbuffer()) == 128 * 1024
    assert not any(ram_file.getbuffer())
    assert isinstance(rtc_file, emulator_module.io.BytesIO)
    assert len(rtc_file.getbuffer()) == 0
    save_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "saved_reset_state"
    )
    input_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "input"
    ]
    assert input_indices
    assert max(input_indices) < save_index

    environment.close()
    assert events[-1] == ("stop", False)
