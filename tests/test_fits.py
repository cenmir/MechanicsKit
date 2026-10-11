"""ISO 286 fits: values checked against Shigley's Tables A-11/A-12 and ISO 286-2."""
import pytest

from mechanicskit.fits import interference, kind, limits, shaft_deviations, tolerance


def test_h7_s6_at_60():
    assert limits('H7/s6', 60) == ((0, 30), (53, 72))
    assert interference('H7/s6', 60) == (23, 72)


def test_clearance_fit_c11_at_34():      # Shigley Example 7-7
    (EI, ES), (ei, es) = limits('H11/c11', 34)
    assert (ES, es, ei) == (160, -120, -280)


def test_size_range_boundaries():        # "over 50 up to and including 65"
    assert shaft_deviations('s6', 65) == (53, 72)
    assert shaft_deviations('s6', 65.1) == (59, 78)


def test_k_only_for_it4_to_it7():
    assert shaft_deviations('k6', 40)[0] == 2
    assert shaft_deviations('k8', 40)[0] == 0


def test_kinds():
    assert kind('H7/g6', 25) == 'clearance'
    assert kind('H7/k6', 25) == 'transition'
    assert kind('H7/u6', 25) == 'interference'


def test_out_of_range():
    with pytest.raises(ValueError):
        tolerance(7, 500)
    with pytest.raises(ValueError):
        limits('G7/h6', 20)


def test_processes_coarsest_first():
    from mechanicskit.fits import processes
    assert processes(7)[0] == 'turning, boring'
    assert processes(6)[0] == 'reaming, fine boring'
    assert 'drilling' not in processes(8)
