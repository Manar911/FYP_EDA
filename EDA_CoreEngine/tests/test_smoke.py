from eda.airport_db import load_airports
from eda.scenario import Scenario, EmergencyType
from eda.validation import validate_scenario, ScenarioValidationError
from eda.features import haversine_km, runway_margin_m, compute_features
from eda.filter import is_feasible


def test_smoke():
    assert 1 + 1 == 2


def test_airports_load():
    airports = load_airports()
    assert len(airports) >= 1
    assert airports[0].icao


def test_scenario_valid_creation():
    s = Scenario(
        aircraft_lat=26.0,
        aircraft_lon=50.0,
        required_runway_m=3000,
        emergency_type=EmergencyType.MEDICAL,
    )
    assert s.emergency_type == EmergencyType.MEDICAL


def test_validate_scenario_passes_for_normal_values():
    s = Scenario(
        aircraft_lat=26.0,
        aircraft_lon=50.0,
        required_runway_m=3000,
        emergency_type=EmergencyType.FUEL,
    )
    validate_scenario(s)


def test_validate_scenario_raises_for_unrealistic_runway():
    s = Scenario(
        aircraft_lat=26.0,
        aircraft_lon=50.0,
        required_runway_m=7000,
        emergency_type=EmergencyType.TECHNICAL,
    )

    try:
        validate_scenario(s)
        assert False, "Expected ScenarioValidationError"
    except ScenarioValidationError as e:
        assert any(issue.field == "required_runway_m" for issue in e.issues)


def test_runway_margin():
    assert runway_margin_m(3000, 4000) == 1000
    assert runway_margin_m(3500, 3400) == -100


def test_haversine_zero_distance():
    d = haversine_km(26.0, 50.0, 26.0, 50.0)
    assert d < 0.001


def test_compute_features_basic():
    airports = load_airports()
    a0 = airports[0]

    s = Scenario(
        aircraft_lat=a0.lat,
        aircraft_lon=a0.lon,
        required_runway_m=3000,
        emergency_type=EmergencyType.MEDICAL,
    )

    f = compute_features(s, a0)
    assert f.distance_km < 0.001
    assert f.runway_margin_m == a0.runway_length_m - 3000


def test_filter_rejects_unreachable_airport():
    airports = load_airports()
    a0 = airports[0]

    s = Scenario(
        aircraft_lat=a0.lat,
        aircraft_lon=a0.lon,
        required_runway_m=3000,
        emergency_type=EmergencyType.FUEL,
    )

    f = compute_features(s, a0)

    # force reject even if distance ~ 0
    r = is_feasible(s, f, max_range_km=-1.0)
    assert r.feasible is False


def test_filter_rejects_short_runway():
    airports = load_airports()
    a0 = airports[0]

    s = Scenario(
        aircraft_lat=a0.lat,
        aircraft_lon=a0.lon,
        required_runway_m=a0.runway_length_m + 500,  # require more than available
        emergency_type=EmergencyType.TECHNICAL,
    )

    f = compute_features(s, a0)
    r = is_feasible(s, f, max_range_km=999999.0)  # ensure range doesn't reject first
    assert r.feasible is False
    assert "runway" in r.reason.lower()


def test_filter_accepts_feasible_airport():
    airports = load_airports()
    a0 = airports[0]

    s = Scenario(
        aircraft_lat=a0.lat,
        aircraft_lon=a0.lon,
        required_runway_m=3000,
        emergency_type=EmergencyType.FUEL,
    )

    f = compute_features(s, a0)
    r = is_feasible(s, f, max_range_km=999999.0)
    assert r.feasible is True