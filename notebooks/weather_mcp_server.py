"""Local MCP weather server used by the notebook examples.

This server keeps the demos deterministic by serving fixed weather data over
MCP stdio transport. It is intended for local testing with LangChain's MCP
adapters rather than production use.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

WEATHER_FIXTURES = {
    "asuncion": {
        "forecast": "Sunny, 31 C",
        "alerts": ["High UV index after 11:00.", "Light breeze from the south."],
    },
    "london": {
        "forecast": "Cloudy, 11 C",
        "alerts": ["Patchy rain in the afternoon.", "Low visibility near the river."],
    },
    "san francisco": {
        "forecast": "Foggy, 16 C",
        "alerts": ["Marine layer until midday.", "Wind picks up after 15:00."],
    },
}

mcp = FastMCP(
    name="weather-demo",
    instructions="Use these tools to answer local weather questions for demo cities.",
)


def _normalize_city(city: str) -> str:
    """Normalize a city string for fixture lookup.

    Args:
        city: Raw city value passed by the caller.

    Returns:
        A normalized lowercase city key.
    """
    return city.strip().lower()


@mcp.tool()
def get_weather(city: str) -> str:
    """Return a deterministic weather summary for a city.

    Args:
        city: The city to look up.

    Returns:
        A short weather summary string.
    """
    city_key = _normalize_city(city)
    fixture = WEATHER_FIXTURES.get(city_key)
    if fixture is None:
        return f"No forecast configured for {city}."
    return fixture["forecast"]


@mcp.tool()
def get_weather_alerts(city: str) -> str:
    """Return a deterministic alert summary for a city.

    Args:
        city: The city to look up.

    Returns:
        A comma-separated string of alerts.
    """
    city_key = _normalize_city(city)
    fixture = WEATHER_FIXTURES.get(city_key)
    if fixture is None:
        return f"No alerts configured for {city}."
    return ", ".join(fixture["alerts"])


if __name__ == "__main__":
    mcp.run("stdio")
