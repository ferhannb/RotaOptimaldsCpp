#pragma once

#include "colreg_types.hpp"

#include <string>
#include <vector>

EncounterScenario make_head_on_scenario();
EncounterScenario make_crossing_starboard_scenario();
EncounterScenario make_crossing_port_scenario();
EncounterScenario make_own_ship_overtaking_scenario();
EncounterScenario make_target_ship_overtaking_scenario();

EncounterScenario make_colreg_scenario_by_name(const std::string& name);
std::vector<std::string> list_standard_colreg_scenario_names();
std::vector<EncounterScenario> make_standard_colreg_scenarios();
