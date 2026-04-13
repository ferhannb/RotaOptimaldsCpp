#include "colreg_classifier.hpp"

#include <cmath>

namespace {
double wrap_to_2pi(double angle) {
  const double wrapped = std::fmod(angle, 2.0 * M_PI);
  return wrapped < 0.0 ? (wrapped + 2.0 * M_PI) : wrapped;
}

double relative_bearing_pm_pi_from(
    const ShipKinematics& reference,
    const ShipKinematics& contact) {
  const double los = std::atan2(contact.y - reference.y, contact.x - reference.x);
  return wrap_to_pi(los - reference.course);
}

double relative_bearing_360_from(
    const ShipKinematics& reference,
    const ShipKinematics& contact) {
  const double los = std::atan2(contact.y - reference.y, contact.x - reference.x);
  return wrap_to_2pi(los - reference.course);
}

double contact_angle_pm_pi(const OwnShip& own_ship, const TargetShip& target_ship) {
  return relative_bearing_pm_pi_from(target_ship.state, own_ship.state);
}

double contact_angle_360(const OwnShip& own_ship, const TargetShip& target_ship) {
  return relative_bearing_360_from(target_ship.state, own_ship.state);
}

bool in_open_interval(double value, double lo, double hi) {
  return value > lo && value < hi;
}
}  // namespace

double deg2rad(double deg) {
  return deg * M_PI / 180.0;
}

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

double wrap_to_pi(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

std::string to_string(EncounterType type) {
  switch (type) {
    case EncounterType::None:
      return "none";
    case EncounterType::HeadOn:
      return "head_on";
    case EncounterType::CrossingStarboard:
      return "crossing_starboard";
    case EncounterType::CrossingPort:
      return "crossing_port";
    case EncounterType::OwnShipOvertaking:
      return "own_ship_overtaking";
    case EncounterType::TargetShipOvertaking:
      return "target_ship_overtaking";
    case EncounterType::SafePassing:
      return "safe_passing";
    case EncounterType::Undefined:
    default:
      return "undefined";
  }
}

std::string to_string(ColregRole role) {
  switch (role) {
    case ColregRole::None:
      return "none";
    case ColregRole::GiveWay:
      return "give_way";
    case ColregRole::StandOn:
      return "stand_on";
    case ColregRole::BothGiveWay:
      return "both_give_way";
    default:
      return "none";
  }
}

RelativePose compute_initial_pose(const OwnShip& own_ship, const TargetShip& target_ship) {
  RelativePose pose;
  pose.relative_bearing = relative_bearing_pm_pi_from(own_ship.state, target_ship.state);
  pose.contact_angle = contact_angle_pm_pi(own_ship, target_ship);
  return pose;
}

CPAInfo compute_cpa(const OwnShip& own_ship, const TargetShip& target_ship) {
  const double rx = target_ship.state.x - own_ship.state.x;
  const double ry = target_ship.state.y - own_ship.state.y;

  const double vox = own_ship.state.speed * std::cos(own_ship.state.course);
  const double voy = own_ship.state.speed * std::sin(own_ship.state.course);
  const double vtx = target_ship.state.speed * std::cos(target_ship.state.course);
  const double vty = target_ship.state.speed * std::sin(target_ship.state.course);

  const double rvx = vtx - vox;
  const double rvy = vty - voy;
  const double rv2 = rvx * rvx + rvy * rvy;

  CPAInfo info;
  info.relative_bearing = relative_bearing_pm_pi_from(own_ship.state, target_ship.state);
  info.course_difference = wrap_to_pi(target_ship.state.course - own_ship.state.course);
  const double range = std::hypot(rx, ry);
  if (range > 1e-12) {
    info.range_rate = ((rx * rvx) + (ry * rvy)) / range;
  }

  if (rv2 < 1e-12) {
    info.tcpa = 0.0;
    info.dcpa = std::hypot(rx, ry);
    info.approaching = false;
    return info;
  }

  info.tcpa = -((rx * rvx) + (ry * rvy)) / rv2;
  const double cx = rx + info.tcpa * rvx;
  const double cy = ry + info.tcpa * rvy;
  info.dcpa = std::hypot(cx, cy);
  info.approaching = (info.tcpa > 0.0) && (info.range_rate < 0.0);
  return info;
}

ColregEncounter classify_encounter(
    const OwnShip& own_ship,
    const TargetShip& target_ship,
    const ColregClassifierConfig& cfg) {
  ColregEncounter out;
  out.initial_pose = compute_initial_pose(own_ship, target_ship);
  out.cpa = compute_cpa(own_ship, target_ship);
  out.collision_risk =
      out.cpa.approaching &&
      out.cpa.tcpa <= cfg.max_tcpa &&
      out.cpa.dcpa <= cfg.risk_dcpa;

  const double beta0_360_deg = rad2deg(relative_bearing_360_from(own_ship.state, target_ship.state));
  const double beta0_pm180_deg = rad2deg(out.initial_pose.relative_bearing);
  const double alpha0_pm180_deg = rad2deg(out.initial_pose.contact_angle);
  const double alpha0_360_deg = rad2deg(contact_angle_360(own_ship, target_ship));
  const bool closing_range = out.cpa.range_rate < 0.0;
  const bool own_faster = own_ship.state.speed > target_ship.state.speed;
  const bool target_faster = target_ship.state.speed > own_ship.state.speed;

  // Paper basis: Woerner et al. Algorithm 3 entry criteria.
  if (in_open_interval(beta0_360_deg, cfg.overtaking_sector_deg, 360.0 - cfg.overtaking_sector_deg) &&
      std::abs(alpha0_pm180_deg) < cfg.alpha_crit_13_deg &&
      closing_range &&
      target_faster) {
    out.type = EncounterType::TargetShipOvertaking;
    out.own_role = ColregRole::StandOn;
    out.rationale = "Rule 13 stand-on geometry: target is abaft own beam, range is closing, and target is the faster vessel.";
    return out;
  }

  if (in_open_interval(alpha0_360_deg, cfg.overtaking_sector_deg, 360.0 - cfg.overtaking_sector_deg) &&
      std::abs(beta0_pm180_deg) < cfg.alpha_crit_13_deg &&
      closing_range &&
      own_faster) {
    out.type = EncounterType::OwnShipOvertaking;
    out.own_role = ColregRole::GiveWay;
    out.rationale = "Rule 13 give-way geometry: own ship is in the target's stern sector, range is closing, and own ship is the faster vessel.";
    return out;
  }

  if (std::abs(beta0_pm180_deg) < cfg.alpha_crit_14_deg &&
      std::abs(alpha0_pm180_deg) < cfg.alpha_crit_14_deg) {
    out.type = EncounterType::HeadOn;
    out.own_role = ColregRole::BothGiveWay;
    out.rationale = "Rule 14 geometry: both relative bearing and contact angle are within the configurable reciprocal-course tolerance.";
    return out;
  }

  if (in_open_interval(beta0_360_deg, 0.0, cfg.crossing_sector_deg) &&
      alpha0_pm180_deg > -cfg.crossing_sector_deg &&
      alpha0_pm180_deg < cfg.alpha_crit_15_deg) {
    out.type = EncounterType::CrossingStarboard;
    out.own_role = ColregRole::GiveWay;
    out.rationale = "Rule 15 give-way geometry: target is on own starboard side and the contact angle is outside head-on/overtaking sectors.";
    return out;
  }

  if (beta0_pm180_deg < 0.0 &&
      std::abs(beta0_pm180_deg) < cfg.crossing_sector_deg &&
      in_open_interval(alpha0_360_deg, 0.0, cfg.crossing_sector_deg)) {
    out.type = EncounterType::CrossingPort;
    out.own_role = ColregRole::StandOn;
    out.rationale = "Rule 15 stand-on geometry: target is on own port side and own ship lies in the target's starboard crossing sector.";
    return out;
  }

  if (!out.collision_risk) {
    out.type = EncounterType::SafePassing;
    out.own_role = ColregRole::None;
    out.rationale = "CPA/TCPA risk thresholds are not exceeded and no COLREG encounter geometry is active.";
    return out;
  }

  out.type = EncounterType::Undefined;
  out.own_role = ColregRole::None;
  out.rationale = "Collision risk exists but encounter geometry does not fit the current deterministic rules.";
  return out;
}
