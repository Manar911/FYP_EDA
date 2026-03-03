from eda.airport_db import load_airports
from eda.scenario import Scenario, EmergencyType
from eda.validation import validate_scenario, ScenarioValidationError


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