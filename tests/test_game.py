"""Contract smoke tests against installed burnysc2, without starting a game."""
from types import SimpleNamespace

import pytest

pytest.importorskip("sc2")
from sc2.ids.unit_typeid import UnitTypeId as U
from sc2.units import Units

from yuri_next.game import YuriBot


def test_observer_uses_modern_api_and_excludes_workers():
    bot = YuriBot()
    bot.state = SimpleNamespace(game_loop=2240)
    bot.units = Units([SimpleNamespace(type_id=U.STALKER, health=40, shield=40,
                                      health_max=80, shield_max=80)], bot)
    bot.enemy_units = Units([SimpleNamespace(tag=1, type_id=U.SCV, can_attack=True),
                            SimpleNamespace(tag=2, type_id=U.MARINE, can_attack=True)], bot)
    bot.townhalls = Units([], bot)
    bot.calculate_supply_cost = lambda kind: 2 if kind == U.STALKER else 1
    bot.minerals, bot.vespene, bot.supply_left = 100, 50, 10
    observed = bot.observe()
    assert observed.army_supply == 2
    assert observed.enemy_army_supply == 1
    assert observed.army_health == 0.5
    assert observed.enemy_visible
    bot.enemy_units = Units([], bot)
    bot.state = SimpleNamespace(game_loop=2464)
    later = bot.observe()
    assert later.enemy_army_supply == 1
    assert later.intel_age_seconds == 10
    assert not later.enemy_visible
