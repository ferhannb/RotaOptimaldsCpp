#pragma once

#include <string>
#include <vector>

struct ShipKinematics {
  double x = 0.0;
  double y = 0.0;
  double course = 0.0;
  double speed = 0.0;
};

struct OwnShip {
  std::string name = "own_ship";
  ShipKinematics state;
  double length = 0.0;
  double beam = 0.0;
};

struct TargetShip {
  std::string id = "target_ship";
  ShipKinematics state;
  double length = 0.0;
  double beam = 0.0;
};

enum class EncounterType {
  None,
  HeadOn,
  CrossingStarboard,
  CrossingPort,
  OwnShipOvertaking,
  TargetShipOvertaking,
  SafePassing,
  Undefined,
};

enum class ColregRole {
  None,
  GiveWay,
  StandOn,
  BothGiveWay,
};

struct CPAInfo {
  double dcpa = 0.0;
  double tcpa = 0.0;
  double relative_bearing = 0.0;
  double course_difference = 0.0;
  double range_rate = 0.0;
  bool approaching = false;
};

struct RelativePose {
  double contact_angle = 0.0;
  double relative_bearing = 0.0;
};

struct ColregEncounter {
  EncounterType type = EncounterType::None;
  ColregRole own_role = ColregRole::None;
  CPAInfo cpa;
  RelativePose initial_pose;
  bool collision_risk = false;
  std::string rationale;
};

struct EncounterScenario {
  std::string name;
  std::string description;
  OwnShip own_ship;
  TargetShip target_ship;
};

double deg2rad(double deg);
double rad2deg(double rad);
double wrap_to_pi(double angle);

std::string to_string(EncounterType type);
std::string to_string(ColregRole role);
