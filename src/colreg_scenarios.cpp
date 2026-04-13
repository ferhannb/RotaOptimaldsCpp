#include "colreg_scenarios.hpp"

#include <stdexcept>

namespace {
OwnShip make_default_own_ship() {
  OwnShip own;
  own.name = "own_ship";
  own.state.x = 0.0;
  own.state.y = 0.0;
  own.state.course = deg2rad(0.0);
  own.state.speed = 6.0;
  own.length = 120.0;
  own.beam = 20.0;
  return own;
}
}  // namespace

EncounterScenario make_head_on_scenario() {
  EncounterScenario scenario;
  scenario.name = "head_on";
  scenario.description = "Both vessels are on nearly reciprocal courses with collision risk ahead.";
  scenario.own_ship = make_default_own_ship();
  scenario.target_ship.id = "target_head_on";
  scenario.target_ship.state.x = 30.0;
  scenario.target_ship.state.y = 0.4;
  scenario.target_ship.state.course = deg2rad(180.0);
  scenario.target_ship.state.speed = 6.0;
  scenario.target_ship.length = 110.0;
  scenario.target_ship.beam = 18.0;
  return scenario;
}

EncounterScenario make_crossing_starboard_scenario() {
  EncounterScenario scenario;
  scenario.name = "crossing_starboard";
  scenario.description = "Target vessel approaches from own ship's starboard side; own ship should give way.";
  scenario.own_ship = make_default_own_ship();
  scenario.target_ship.id = "target_starboard";
  scenario.target_ship.state.x = 18.0;
  scenario.target_ship.state.y = 18.0;
  scenario.target_ship.state.course = deg2rad(270.0);
  scenario.target_ship.state.speed = 6.0;
  scenario.target_ship.length = 90.0;
  scenario.target_ship.beam = 16.0;
  return scenario;
}

EncounterScenario make_crossing_port_scenario() {
  EncounterScenario scenario;
  scenario.name = "crossing_port";
  scenario.description = "Target vessel approaches from own ship's port side; own ship should stand on.";
  scenario.own_ship = make_default_own_ship();
  scenario.target_ship.id = "target_port";
  scenario.target_ship.state.x = 18.0;
  scenario.target_ship.state.y = -18.0;
  scenario.target_ship.state.course = deg2rad(90.0);
  scenario.target_ship.state.speed = 6.0;
  scenario.target_ship.length = 90.0;
  scenario.target_ship.beam = 16.0;
  return scenario;
}

EncounterScenario make_own_ship_overtaking_scenario() {
  EncounterScenario scenario;
  scenario.name = "own_ship_overtaking";
  scenario.description = "Own ship is faster and approaches the target from abaft the beam.";
  scenario.own_ship = make_default_own_ship();
  scenario.own_ship.state.speed = 8.0;
  scenario.target_ship.id = "target_overtaken";
  scenario.target_ship.state.x = 20.0;
  scenario.target_ship.state.y = 0.4;
  scenario.target_ship.state.course = deg2rad(0.0);
  scenario.target_ship.state.speed = 4.0;
  scenario.target_ship.length = 80.0;
  scenario.target_ship.beam = 14.0;
  return scenario;
}

EncounterScenario make_target_ship_overtaking_scenario() {
  EncounterScenario scenario;
  scenario.name = "target_ship_overtaking";
  scenario.description = "Target ship is faster and approaches own ship from abaft the beam.";
  scenario.own_ship = make_default_own_ship();
  scenario.own_ship.state.speed = 4.0;
  scenario.target_ship.id = "target_overtaking";
  scenario.target_ship.state.x = -20.0;
  scenario.target_ship.state.y = 0.4;
  scenario.target_ship.state.course = deg2rad(0.0);
  scenario.target_ship.state.speed = 8.0;
  scenario.target_ship.length = 80.0;
  scenario.target_ship.beam = 14.0;
  return scenario;
}

std::vector<EncounterScenario> make_standard_colreg_scenarios() {
  return {
      make_head_on_scenario(),
      make_crossing_starboard_scenario(),
      make_crossing_port_scenario(),
      make_own_ship_overtaking_scenario(),
      make_target_ship_overtaking_scenario(),
  };
}

EncounterScenario make_colreg_scenario_by_name(const std::string& name) {
  if (name == "head_on") {
    return make_head_on_scenario();
  }
  if (name == "crossing_starboard") {
    return make_crossing_starboard_scenario();
  }
  if (name == "crossing_port") {
    return make_crossing_port_scenario();
  }
  if (name == "own_ship_overtaking" || name == "overtaking_own") {
    return make_own_ship_overtaking_scenario();
  }
  if (name == "target_ship_overtaking" || name == "overtaken_by_target") {
    return make_target_ship_overtaking_scenario();
  }
  throw std::runtime_error("Unknown COLREG preset scenario: " + name);
}

std::vector<std::string> list_standard_colreg_scenario_names() {
  return {
      "head_on",
      "crossing_starboard",
      "crossing_port",
      "own_ship_overtaking",
      "target_ship_overtaking",
  };
}
