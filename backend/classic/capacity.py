"""Room capacity maths for fixed-size cartons.

Default carton (centimetres): height 25, width 30, base (depth) 35.
All room inputs are in metres.
"""
from dataclasses import dataclass
import math

CM_PER_M = 100.0


@dataclass
class BoxSize:
    height: float = 25.0   # cm (vertical)
    width: float = 30.0    # cm
    base: float = 35.0     # cm (front-to-back depth)

    @property
    def footprint_cm2(self) -> float:
        return self.width * self.base


def grid_fit(length_cm: float, breadth_cm: float, w: float, d: float) -> int:
    """Most cartons that fit on a floor as a straight grid; tries both orientations."""
    a = int(length_cm // w) * int(breadth_cm // d)
    b = int(length_cm // d) * int(breadth_cm // w)
    return max(a, b)


def room_capacity(room_length_m: float, room_width_m: float, max_stack_height_m: float,
                  box: BoxSize = BoxSize(), usable_fraction: float = 0.6) -> dict:
    """Expected number of cartons the room can hold.

    usable_fraction: share of the floor that can hold cartons (the rest is aisles,
    walkways, doors, machines). 0.6 means 60 % of the floor is usable.
    """
    L, W = room_length_m * CM_PER_M, room_width_m * CM_PER_M
    floor_area_m2 = room_length_m * room_width_m
    full_grid = grid_fit(L, W, box.width, box.base)            # no aisles at all
    per_layer = int(full_grid * usable_fraction)               # after aisles
    layers = int((max_stack_height_m * CM_PER_M) // box.height)
    total = per_layer * layers
    return {
        "floor_area_m2": floor_area_m2,
        "usable_area_m2": floor_area_m2 * usable_fraction,
        "box_footprint_cm2": box.footprint_cm2,
        "boxes_per_layer": per_layer,
        "layers": layers,
        "capacity": total,
    }


def fill_status(detected: int, capacity: int) -> dict:
    pct = 100.0 * detected / capacity if capacity else 0.0
    return {"detected": detected, "capacity": capacity,
            "fill_pct": pct, "remaining": max(capacity - detected, 0)}
