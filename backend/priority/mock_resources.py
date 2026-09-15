"""
AURA Priority Engine – Mock Resource Pool
==========================================

⚠️  DEMO DATA ONLY ⚠️
All resources below are FICTIONAL and created for the college demo.
They do NOT reflect real-time availability, actual distances, or
real hospital/ambulance/rescue-crew data.

──────────────────────────────────────────────────────────────────────────────

Resource selection strategy (demo):
  - Each resource pool is stored as a list.
  - select_resource() picks the nearest AVAILABLE resource.
  - If none are available, the closest unavailable one is returned with a flag.
"""

import copy
import random
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Mock data pools
# ──────────────────────────────────────────────────────────────────────────────

_AMBULANCES = [
    {
        "id": "AMB-01",
        "name": "City General Ambulance – Unit 1",
        "availability": "available",
        "distance_km": 1.2,
        "estimated_response_minutes": 4,
        "contact": "112",
    },
    {
        "id": "AMB-02",
        "name": "St. Mary's Ambulance – Unit 7",
        "availability": "available",
        "distance_km": 2.8,
        "estimated_response_minutes": 8,
        "contact": "112",
    },
    {
        "id": "AMB-03",
        "name": "Metro EMS – Unit 3",
        "availability": "busy",
        "distance_km": 3.5,
        "estimated_response_minutes": 15,
        "contact": "112",
    },
    {
        "id": "AMB-04",
        "name": "North District Ambulance – Unit 2",
        "availability": "available",
        "distance_km": 5.1,
        "estimated_response_minutes": 14,
        "contact": "112",
    },
]

_HOSPITALS = [
    {
        "id": "HSP-01",
        "name": "City General Hospital",
        "availability": "open",
        "distance_km": 1.5,
        "estimated_response_minutes": 5,   # transfer time
        "speciality": "Emergency & Trauma",
        "beds_available": 12,
    },
    {
        "id": "HSP-02",
        "name": "St. Mary's Medical Centre",
        "availability": "open",
        "distance_km": 3.0,
        "estimated_response_minutes": 9,
        "speciality": "General Emergency",
        "beds_available": 6,
    },
    {
        "id": "HSP-03",
        "name": "Apollo Trauma Centre",
        "availability": "open",
        "distance_km": 4.4,
        "estimated_response_minutes": 13,
        "speciality": "Trauma & Burns",
        "beds_available": 20,
    },
    {
        "id": "HSP-04",
        "name": "North District Community Hospital",
        "availability": "limited",
        "distance_km": 6.2,
        "estimated_response_minutes": 18,
        "speciality": "General",
        "beds_available": 2,
    },
]

_RESCUE_CREWS = [
    {
        "id": "RSC-01",
        "name": "Urban Rescue Team Alpha",
        "availability": "available",
        "distance_km": 2.0,
        "estimated_response_minutes": 7,
        "speciality": "Flood & Water Rescue",
        "crew_size": 6,
    },
    {
        "id": "RSC-02",
        "name": "Fire Brigade – Rescue Unit 5",
        "availability": "available",
        "distance_km": 2.7,
        "estimated_response_minutes": 9,
        "speciality": "Vehicle Extrication & Fire",
        "crew_size": 8,
    },
    {
        "id": "RSC-03",
        "name": "NDRF Quick Response Team",
        "availability": "standby",
        "distance_km": 8.3,
        "estimated_response_minutes": 22,
        "speciality": "Disaster Relief",
        "crew_size": 12,
    },
    {
        "id": "RSC-04",
        "name": "Civil Defence – Unit 2",
        "availability": "available",
        "distance_km": 4.6,
        "estimated_response_minutes": 14,
        "speciality": "General Rescue",
        "crew_size": 5,
    },
]

# Maps resource type → pool list
_POOLS = {
    "ambulance": _AMBULANCES,
    "hospital": _HOSPITALS,
    "rescue_crew": _RESCUE_CREWS,
}

# Available status labels (in priority order for selection)
_AVAILABLE_STATUSES = {"available", "open", "standby"}


# ──────────────────────────────────────────────────────────────────────────────
# Selection logic
# ──────────────────────────────────────────────────────────────────────────────

def select_resource(resource_type: str) -> dict:
    """
    Return the nearest *available* resource of the given type.

    Falls back to the nearest resource regardless of availability if none
    are currently free.

    Parameters
    ----------
    resource_type : str   One of "ambulance", "hospital", "rescue_crew"

    Returns
    -------
    dict   A deep copy of the selected resource entry, enriched with a
           ``demo_data`` disclaimer field.
    """
    pool = _POOLS.get(resource_type, [])
    if not pool:
        return _unavailable_placeholder(resource_type)

    # Sort by distance; prefer available over unavailable
    available = [r for r in pool if r.get("availability") in _AVAILABLE_STATUSES]
    candidates = available if available else pool

    selected = min(candidates, key=lambda r: r["distance_km"])

    result = copy.deepcopy(selected)
    result["demo_data"] = True   # ← explicit flag: this is mock data
    return result


def select_all_resources(resource_type: str) -> list:
    """
    Return all resources of the given type, sorted by distance.
    Useful for showing a full resource list in the UI.
    """
    pool = _POOLS.get(resource_type, [])
    sorted_pool = sorted(pool, key=lambda r: r["distance_km"])
    return [
        {**copy.deepcopy(r), "demo_data": True}
        for r in sorted_pool
    ]


def _unavailable_placeholder(resource_type: str) -> dict:
    return {
        "id": None,
        "name": f"No {resource_type.replace('_', ' ')} data available",
        "availability": "unknown",
        "distance_km": None,
        "estimated_response_minutes": None,
        "demo_data": True,
    }
