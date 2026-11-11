"""Entry point for the EV project."""


def estimate_range_km(battery_kwh: float, efficiency_kwh_per_100km: float) -> float:
    """Estimate driving range from battery size and energy use."""
    if efficiency_kwh_per_100km <= 0:
        raise ValueError("efficiency must be positive")
    return battery_kwh / efficiency_kwh_per_100km * 100


def main() -> None:
    battery = 60.0      # kWh
    efficiency = 15.0   # kWh per 100 km
    print(f"Estimated range: {estimate_range_km(battery, efficiency):.0f} km")


if __name__ == "__main__":
    main()
