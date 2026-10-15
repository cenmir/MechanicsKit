"""Friction coefficients: values checked against the sources named in mechanicskit.friction."""
import pytest

from mechanicskit import friction


def test_marks_pairs():                  # NASA RP-1228 Table IV, from Marks 8th ed.
    assert friction.pair('mild steel', 'mild steel')['static_dry'] == 0.74
    assert friction.pair('mild steel', 'mild steel')['sliding_dry'] == 0.57
    assert friction.pair('hard steel', 'hard steel')['static_dry'] == 0.78
    assert friction.pair('cast iron', 'cast iron')['sliding_dry'] == 0.15


def test_every_pair_has_both_dry_values():
    for d in friction.PAIRS.values():
        assert d['static_dry'] is not None and d['sliding_dry'] is not None


def test_pair_either_order():
    assert friction.pair('Mild steel', 'Aluminium') == friction.pair('aluminium', 'mild steel')
    with pytest.raises(KeyError):
        friction.pair('mild steel', 'cast iron')     # only the sliding value is known


def test_vdi2230_classes():
    assert friction.bolt_class('b') == (0.08, 0.16)
    assert friction.bolt_class('E') == (0.30, None)
    assert friction.classify_bolt(0.12) == ['B']
    assert friction.classify_bolt(0.15) == ['B', 'C']
    assert friction.classify_bolt(0.50) == ['E']


def test_linings_match_shigley_16_3():
    assert friction.LININGS['woven cotton'][0] == 0.47
    assert friction.LININGS['resilient paper, wet'][0] == (0.09, 0.15)


def test_pressed_fit():                  # DIN 7190, S235 hub
    assert friction.pressed_fit('S235') == {'hold': 0.10, 'press': 0.09}
    assert friction.pressed_fit('S235', lubricated=True) == {'hold': 0.07, 'press': 0.06}


def test_table_markdown():
    t = str(friction.table('pairs', source=False))
    assert t.splitlines()[0].startswith('| Surfaces |')
    assert '| Mild steel on mild steel | 0.74 |  | 0.57 | 0.09–0.19 |' in t
    assert '≥ 0.30' in str(friction.table('bolts'))
    assert 'Source:' in str(friction.table('linings'))
    with pytest.raises(KeyError):
        friction.table('ice')


def test_materials_and_partners():
    assert 'Oak' in friction.materials()
    assert friction.partners('Oak') == ['Oak', 'Leather']
    for m in friction.materials():                   # every partner gives a pair
        for p in friction.partners(m):
            friction.pair(m, p)
