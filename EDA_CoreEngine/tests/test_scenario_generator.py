from eda.aircraft_db import load_aircraft_profiles
from eda.scenario_generator import (
    ScenarioGenerator,
    GeneratedScenario,
    EMERGENCY_RANGE_KM,
    CRITICAL_EMERGENCIES,
    NON_CRITICAL_EMERGENCIES,
)


def test_generate_one_returns_generated_scenario():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)

    assert isinstance(scenario, GeneratedScenario)
    assert scenario.scenario_id == "S00001"
    assert isinstance(scenario.aircraft_lat, float)
    assert isinstance(scenario.aircraft_lon, float)
    assert isinstance(scenario.aircraft_type, str)
    assert isinstance(scenario.aircraft_category, str)
    assert isinstance(scenario.required_runway_m, int)
    assert isinstance(scenario.emergency_type, str)
    assert isinstance(scenario.max_range_km, int)
    assert isinstance(scenario.seed_airport_icao, str)


def test_generate_one_uses_valid_aircraft_profile():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)

    aircraft_profiles = load_aircraft_profiles()

    matching_profiles = [
        p for p in aircraft_profiles
        if p.aircraft_type == scenario.aircraft_type
        and p.aircraft_category == scenario.aircraft_category
    ]

    assert len(matching_profiles) == 1


def test_required_runway_is_within_selected_aircraft_policy():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)

    aircraft_profiles = load_aircraft_profiles()

    profile = next(
        p for p in aircraft_profiles
        if p.aircraft_type == scenario.aircraft_type
        and p.aircraft_category == scenario.aircraft_category
    )

    assert profile.runway_min_m <= scenario.required_runway_m <= profile.runway_max_m


def test_emergency_type_is_valid():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)

    assert scenario.emergency_type in EMERGENCY_RANGE_KM


def test_max_range_matches_emergency_policy():
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_many(count=20, start_index=1)

    for scenario in scenarios:
        min_km, max_km = EMERGENCY_RANGE_KM[scenario.emergency_type]
        assert min_km <= scenario.max_range_km <= max_km


def test_seed_airport_icao_exists_in_loaded_airports():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)

    airport_icaos = {airport.icao for airport in generator.airports}
    assert scenario.seed_airport_icao in airport_icaos


def test_generate_many_returns_requested_count():
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_many(count=5, start_index=1)

    assert len(scenarios) == 5


def test_generate_many_assigns_unique_scenario_ids():
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_many(count=10, start_index=1)

    scenario_ids = [s.scenario_id for s in scenarios]
    assert len(scenario_ids) == len(set(scenario_ids))


def test_generate_many_assigns_sequential_ids():
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_many(count=3, start_index=7)

    ids = [s.scenario_id for s in scenarios]
    assert ids == ["S00007", "S00008", "S00009"]


def test_generate_one_with_known_template_balanced():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1, template="balanced")

    assert scenario.scenario_id == "S00001"
    assert scenario.required_runway_m > 0
    assert scenario.max_range_km > 0


def test_generate_one_with_known_template_short_range():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1, template="short_range")

    assert scenario.scenario_id == "S00001"
    assert scenario.required_runway_m > 0
    assert scenario.max_range_km > 0


def test_generate_one_with_known_template_tight_runway():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1, template="tight_runway")

    assert scenario.scenario_id == "S00001"
    assert scenario.required_runway_m > 0
    assert scenario.max_range_km > 0


def test_generate_one_with_known_template_competitive():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1, template="competitive")

    assert scenario.scenario_id == "S00001"
    assert scenario.required_runway_m > 0
    assert scenario.max_range_km > 0


def test_generate_one_with_known_template_critical_pressure():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1, template="critical_pressure")

    assert scenario.scenario_id == "S00001"
    assert scenario.emergency_type in CRITICAL_EMERGENCIES


def test_invalid_template_raises_value_error():
    generator = ScenarioGenerator(seed=42)

    try:
        generator.generate_one(scenario_index=1, template="not_a_real_template")
        assert False, "Expected ValueError for invalid template"
    except ValueError as e:
        assert "Unknown template" in str(e)


def test_generate_many_with_templates_returns_matching_count():
    generator = ScenarioGenerator(seed=42)
    templates = ["balanced", "short_range", "tight_runway", "competitive"]

    scenarios = generator.generate_many(count=4, start_index=1, templates=templates)

    assert len(scenarios) == 4
    assert scenarios[0].scenario_id == "S00001"
    assert scenarios[1].scenario_id == "S00002"
    assert scenarios[2].scenario_id == "S00003"
    assert scenarios[3].scenario_id == "S00004"


def test_generate_many_with_wrong_template_count_raises():
    generator = ScenarioGenerator(seed=42)

    try:
        generator.generate_many(
            count=3,
            start_index=1,
            templates=["balanced", "short_range"],
        )
        assert False, "Expected ValueError when templates length != count"
    except ValueError as e:
        assert "length must equal count" in str(e)


def test_generator_is_reproducible_with_same_seed():
    generator1 = ScenarioGenerator(seed=123)
    generator2 = ScenarioGenerator(seed=123)

    scenarios1 = generator1.generate_many(count=5, start_index=1)
    scenarios2 = generator2.generate_many(count=5, start_index=1)

    dicts1 = [s.to_dict() for s in scenarios1]
    dicts2 = [s.to_dict() for s in scenarios2]

    assert dicts1 == dicts2


def test_to_dict_contains_expected_keys():
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate_one(scenario_index=1)
    data = scenario.to_dict()

    expected_keys = {
        "scenario_id",
        "aircraft_lat",
        "aircraft_lon",
        "aircraft_type",
        "aircraft_category",
        "required_runway_m",
        "emergency_type",
        "max_range_km",
        "seed_airport_icao",
    }

    assert set(data.keys()) == expected_keys


def test_emergency_groups_are_disjoint():
    assert CRITICAL_EMERGENCIES.isdisjoint(NON_CRITICAL_EMERGENCIES)


def test_all_emergency_types_are_covered_by_groups():
    grouped = CRITICAL_EMERGENCIES | NON_CRITICAL_EMERGENCIES
    configured = set(EMERGENCY_RANGE_KM.keys())

    assert grouped == configured