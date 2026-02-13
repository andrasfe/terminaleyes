"""Tests for USB HID scan code mappings."""

from __future__ import annotations

import pytest

from terminaleyes.raspi.hid_codes import (
    KEY_CODES,
    MODIFIER_LEFT_ALT,
    MODIFIER_LEFT_CTRL,
    MODIFIER_LEFT_META,
    MODIFIER_LEFT_SHIFT,
    MODIFIER_MAP,
    MODIFIER_NONE,
    SHIFT_CHARS,
    char_to_hid,
    key_name_to_hid,
    modifiers_to_bitmask,
)


class TestKeyCodes:
    def test_letters_are_contiguous(self) -> None:
        for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
            assert KEY_CODES[letter] == 0x04 + i

    def test_digits_are_contiguous(self) -> None:
        for i, digit in enumerate("1234567890"):
            assert KEY_CODES[digit] == 0x1E + i

    def test_common_special_keys_present(self) -> None:
        assert KEY_CODES["Enter"] == 0x28
        assert KEY_CODES["Tab"] == 0x2B
        assert KEY_CODES["Space"] == 0x2C
        assert KEY_CODES["Escape"] == 0x29
        assert KEY_CODES["Backspace"] == 0x2A
        assert KEY_CODES["Delete"] == 0x4C

    def test_arrow_keys(self) -> None:
        assert KEY_CODES["Up"] == 0x52
        assert KEY_CODES["Down"] == 0x51
        assert KEY_CODES["Left"] == 0x50
        assert KEY_CODES["Right"] == 0x4F

    def test_function_keys(self) -> None:
        for i in range(1, 13):
            assert f"F{i}" in KEY_CODES


class TestModifiers:
    def test_ctrl_maps(self) -> None:
        assert MODIFIER_MAP["ctrl"] == MODIFIER_LEFT_CTRL
        assert MODIFIER_MAP["left_ctrl"] == MODIFIER_LEFT_CTRL

    def test_shift_maps(self) -> None:
        assert MODIFIER_MAP["shift"] == MODIFIER_LEFT_SHIFT

    def test_alt_maps(self) -> None:
        assert MODIFIER_MAP["alt"] == MODIFIER_LEFT_ALT

    def test_meta_aliases(self) -> None:
        assert MODIFIER_MAP["meta"] == MODIFIER_LEFT_META
        assert MODIFIER_MAP["super"] == MODIFIER_LEFT_META
        assert MODIFIER_MAP["win"] == MODIFIER_LEFT_META


class TestCharToHid:
    def test_lowercase_letter(self) -> None:
        mod, code = char_to_hid("a")
        assert mod == MODIFIER_NONE
        assert code == 0x04

    def test_uppercase_letter(self) -> None:
        mod, code = char_to_hid("A")
        assert mod == MODIFIER_LEFT_SHIFT
        assert code == 0x04

    def test_digit(self) -> None:
        mod, code = char_to_hid("5")
        assert mod == MODIFIER_NONE
        assert code == 0x22

    def test_shifted_symbol(self) -> None:
        mod, code = char_to_hid("!")
        assert mod == MODIFIER_LEFT_SHIFT
        assert code == KEY_CODES["1"]

    def test_space(self) -> None:
        mod, code = char_to_hid(" ")
        assert mod == MODIFIER_NONE
        assert code == 0x2C

    def test_unknown_char_raises(self) -> None:
        with pytest.raises(ValueError, match="No HID mapping"):
            char_to_hid("\x00")


class TestKeyNameToHid:
    def test_named_key(self) -> None:
        assert key_name_to_hid("Enter") == 0x28

    def test_single_char_key(self) -> None:
        assert key_name_to_hid("a") == 0x04

    def test_case_insensitive_single_char(self) -> None:
        # 'A' falls through to case-insensitive lookup
        assert key_name_to_hid("A") == 0x04

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown key name"):
            key_name_to_hid("NonExistentKey")


class TestModifiersToBitmask:
    def test_single_modifier(self) -> None:
        assert modifiers_to_bitmask(["ctrl"]) == MODIFIER_LEFT_CTRL

    def test_multiple_modifiers(self) -> None:
        result = modifiers_to_bitmask(["ctrl", "shift"])
        assert result == (MODIFIER_LEFT_CTRL | MODIFIER_LEFT_SHIFT)

    def test_empty_list(self) -> None:
        assert modifiers_to_bitmask([]) == MODIFIER_NONE

    def test_unknown_modifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown modifier"):
            modifiers_to_bitmask(["banana"])


class TestShiftChars:
    def test_all_shift_chars_have_base_mapping(self) -> None:
        """Every shifted char should map to a base char in KEY_CODES."""
        for shifted, base in SHIFT_CHARS.items():
            assert base in KEY_CODES, f"Shift char {shifted!r} -> {base!r} not in KEY_CODES"
