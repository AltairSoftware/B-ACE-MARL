"""
Rule-based red team policy: chase the highest-priority detected target and
fire missiles probabilistically, mirroring the baseline1 FSM behaviour.

Observation structure (from Fighter.gd get_obs()):
    [0]  own_x_pos
    [1]  own_z_pos
    [2]  own_altitude          = global_y_gdm / 150.0
    [3]  own_dist_target
    [4]  own_aspect_angle_target
    [5]  own_current_hdg       = current_hdg_deg / 180   (-1..1)
    [6]  own_current_speed
    [7]  own_missiles
    [8]  own_in_flight_missile
    then for each tracked enemy (13 values each):
        track_alt_diff_{id}
        track_aspect_angle_{id}   <- bearing error, already in [-1, 1]
        track_angle_off_{id}
        track_dist_{id}           <- -1.0 when track is dead (sentinel)
        track_dist2go_{id}
        track_own_missile_RMax_{id}
        track_own_missile_Nez_{id}
        track_enemy_missile_RMax_{id}
        track_enemy_missile_Nez_{id}
        track_threat_factor_{id}
        track_offensive_factor_{id}  <- stored as (offensive_factor - 1)
        track_is_missile_support_{id}
        track_detected_{id}       <- 1.0 if currently on radar

Action space (Low_Level_Continuous):
    [0] heading  (-1..1) -> ±180° relative turn from current heading
    [1] altitude (-1..1) -> 0..50000 ft
    [2] desired_g (-1..1)
    [3] fire      (>0 fires, Godot still gates on aspect angle < 30°)

Conversion constants (from Godot Sim_assets.gd):
    FT2GDM   = 0.3048 / 100.0
    NM2GDM   = 1852.0 / 100.0
    GRAVITY_GDM = 9.81 / 100.0

Heading coordinate system (from Calc.get_hdg_2d = atan2(dx, -dz)):
    0°  = north (-z), 90° = east (+x), ±180° = south (+z), -90° = west (-x)
"""

import numpy as np


class RedTeamPolicy:
    # obs[own_altitude] = global_y_gdm / 150; action[alt] = alt_ft/25000 - 1
    # alt_ft = global_y_gdm / FT2GDM; FT2GDM = 0.3048/100
    _ALT_SCALE  = 150.0 * 100.0 / (0.3048 * 25000.0)  # ≈ 1.9685

    # NM -> normalised obs coordinate (obs = global_gdm / 3000)
    _NM_TO_NORM = 1852.0 / 100.0 / 3000.0              # ≈ 0.006173

    def __init__(
        self,
        obs_maps: dict,
        shot_threshold: float = 0.85,
        shot_variation: float = 0.10,
        chase_g: float = 0.5,
        combat_area: dict | None = None,
        boundary_margin_nm: float = 0.2,
        seed: int | None = None,
    ):
        """
        Args:
            obs_maps:            {agent_name: {label: obs_index}} for every red agent.
            shot_threshold:      Fire when offensive_factor exceeds this value.
            shot_variation:      ±uniform noise on shot_threshold (mirrors baseline1).
            chase_g:             Normalised G command while turning toward target.
            combat_area:         Dict with x_min/x_max/z_min/z_max in NM, or None.
            boundary_margin_nm:  Distance from wall (NM) where avoidance kicks in.
            seed:                RNG seed for reproducibility.
        """
        self.obs_maps            = obs_maps
        self.shot_threshold      = shot_threshold
        self.shot_variation      = shot_variation
        self.chase_g             = chase_g
        self.combat_area         = combat_area
        self.boundary_margin_nm  = boundary_margin_nm
        self._rng                = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, agent_name: str, obs: np.ndarray) -> np.ndarray:
        """Return a 4-D action for one red agent given its raw observation."""
        labels  = self.obs_maps[agent_name]
        own_alt = float(obs[labels["own_altitude"]])
        alt_cmd = float(np.clip(own_alt * self._ALT_SCALE - 1.0, -1.0, 1.0))

        # --- Chase target ---
        target = self._best_target(obs, labels)
        if target is not None:
            aspect, offensive, is_detected = target
            chase_hdg = float(np.clip(aspect, -1.0, 1.0))
            g_cmd = (
                float(np.clip(abs(aspect) * 2.0, 0.0, 1.0)) * self.chase_g
                + (1.0 - self.chase_g) * 0.3
            )
            fire_cmd = 0.0
            if is_detected:
                thr = self.shot_threshold + self._rng.uniform(
                    -self.shot_variation, self.shot_variation
                )
                fire_cmd = 1.0 if offensive > thr else 0.0
        else:
            chase_hdg = 0.0
            g_cmd     = 0.3
            fire_cmd  = 0.0

        # --- Boundary avoidance ---
        # Blends heading toward area centre when approaching a wall.
        # Strength goes from 0 (at margin distance) to 1 (at the wall).
        boundary_hdg, strength = self._boundary_heading(obs, labels)
        heading_cmd = float(np.clip(
            (1.0 - strength) * chase_hdg + strength * boundary_hdg,
            -1.0, 1.0,
        ))

        # Suppress fire while correcting strongly toward a boundary
        if strength > 0.5:
            fire_cmd = 0.0

        return np.array([heading_cmd, alt_cmd, g_cmd, fire_cmd], dtype=np.float32)

    def act_batch(
        self, agent_names: list[str], obs_array: np.ndarray
    ) -> np.ndarray:
        """
        Compute actions for multiple red agents in one call.

        Args:
            agent_names: Ordered list of red agent names (length n_red).
            obs_array:   Observations, shape (n_red, obs_dim).

        Returns:
            actions of shape (n_red, 4).
        """
        return np.stack(
            [self.act(name, obs_array[i]) for i, name in enumerate(agent_names)],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_target(self, obs: np.ndarray, labels: dict):
        """
        Find the highest-priority alive target.

        Key insight: Track.update_track() computes aspect_angle from the ACTUAL
        current position unconditionally — before the radar-FOV check.
        So track_aspect_angle_{id} is geometrically accurate even when
        track_detected_{id} = 0 (target is outside the ±60° radar cone).
        The dead-track sentinel is track_dist_{id} = -1.0.

        Priority: detected > undetected-but-alive, then by offensive_factor.

        Returns (aspect_angle_normalised, offensive_factor, is_detected) or None.
        """
        best_aspect    = None
        best_offensive = -np.inf
        best_detected  = False

        for label, idx in labels.items():
            if not label.startswith("track_dist_"):
                continue
            if float(obs[idx]) < 0:      # -1.0 sentinel → track is dead
                continue

            track_id      = label[len("track_dist_"):]
            aspect_key    = f"track_aspect_angle_{track_id}"
            offensive_key = f"track_offensive_factor_{track_id}"
            detected_key  = f"track_detected_{track_id}"

            if aspect_key not in labels:
                continue

            offensive   = float(obs[labels[offensive_key]]) + 1.0 if offensive_key in labels else 0.0
            is_detected = float(obs[labels[detected_key]]) > 0.5   if detected_key  in labels else False

            if (int(is_detected), offensive) > (int(best_detected), best_offensive):
                best_aspect    = float(obs[labels[aspect_key]])
                best_offensive = offensive
                best_detected  = is_detected

        if best_aspect is None:
            return None
        return best_aspect, best_offensive, best_detected

    def _boundary_heading(self, obs: np.ndarray, labels: dict) -> tuple[float, float]:
        """
        Compute a heading correction that steers toward the combat-area centre
        when the agent is within boundary_margin_nm of any wall.

        Coordinate system (Godot / Calc.get_hdg_2d):
            heading 0°  = north (-z direction)
            heading 90° = east  (+x direction)
            own_x_pos  = global_x / 3000  (positive = east)
            own_z_pos  = global_z / 3000  (positive = south)
            own_current_hdg = current_hdg_deg / 180  ∈ [-1, 1]

        Returns:
            (boundary_heading_cmd, strength)
            boundary_heading_cmd : relative heading offset in [-1, 1]
            strength             : blend weight 0 (far) → 1 (at wall)
        """
        if self.combat_area is None:
            return 0.0, 0.0

        x       = float(obs[labels["own_x_pos"]])
        z       = float(obs[labels["own_z_pos"]])
        hdg_deg = float(obs[labels["own_current_hdg"]]) * 180.0  # degrees

        k      = self._NM_TO_NORM
        margin = self.boundary_margin_nm * k

        ca    = self.combat_area
        x_min = ca["x_min"] * k;  x_max = ca["x_max"] * k
        z_min = ca["z_min"] * k;  z_max = ca["z_max"] * k

        # Distance from agent to each wall (positive = inside)
        dist_x_min = x - x_min
        dist_x_max = x_max - x
        dist_z_min = z - z_min
        dist_z_max = z_max - z
        min_dist   = min(dist_x_min, dist_x_max, dist_z_min, dist_z_max)

        if min_dist >= margin:
            return 0.0, 0.0

        # Blend strength: 1 at the wall, 0 at margin distance
        strength = float(np.clip(1.0 - min_dist / margin, 0.0, 1.0))

        # Desired heading: toward the centre of the combat area
        x_ctr = (x_min + x_max) / 2.0
        z_ctr = (z_min + z_max) / 2.0
        dx = x_ctr - x
        dz = z_ctr - z

        # atan2(dx, -dz) matches Calc.get_hdg_2d convention
        desired_hdg_deg = float(np.degrees(np.arctan2(dx, -dz)))

        # Relative heading offset from current heading, wrapped to [-180, 180]
        hdg_diff = (desired_hdg_deg - hdg_deg + 180.0) % 360.0 - 180.0
        boundary_cmd = float(np.clip(hdg_diff / 180.0, -1.0, 1.0))

        return boundary_cmd, strength
