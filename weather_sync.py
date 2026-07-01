"""
Weather Sync - Real weather for real worlds.

"If it's raining in Denver, it's raining in the apocalypse."

Uses OpenWeatherMap API (free tier: 1000 calls/day).
Get your API key at: https://openweathermap.org/api

The sync is gentle - caches for 30 minutes, doesn't spam the API.
Weather affects mood, NPC behavior, and narrative texture.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import debug_logger as dbg


# Weather condition mappings (OpenWeatherMap -> game weather)
WEATHER_MAP = {
    # Clear
    "clear": "clear",
    "clear sky": "clear",

    # Clouds
    "few clouds": "partly_cloudy",
    "scattered clouds": "cloudy",
    "broken clouds": "overcast",
    "overcast clouds": "overcast",

    # Rain
    "light rain": "light_rain",
    "moderate rain": "rain",
    "heavy intensity rain": "heavy_rain",
    "very heavy rain": "storm",
    "extreme rain": "storm",
    "freezing rain": "freezing_rain",
    "shower rain": "rain",
    "light intensity shower rain": "light_rain",
    "heavy intensity shower rain": "heavy_rain",

    # Drizzle
    "light intensity drizzle": "drizzle",
    "drizzle": "drizzle",
    "heavy intensity drizzle": "light_rain",

    # Thunderstorm
    "thunderstorm": "thunderstorm",
    "thunderstorm with light rain": "thunderstorm",
    "thunderstorm with rain": "thunderstorm",
    "thunderstorm with heavy rain": "thunderstorm",
    "light thunderstorm": "thunderstorm",
    "heavy thunderstorm": "severe_thunderstorm",
    "ragged thunderstorm": "thunderstorm",

    # Snow
    "light snow": "light_snow",
    "snow": "snow",
    "heavy snow": "heavy_snow",
    "sleet": "sleet",
    "light shower sleet": "sleet",
    "shower sleet": "sleet",
    "light rain and snow": "sleet",
    "rain and snow": "sleet",
    "light shower snow": "light_snow",
    "shower snow": "snow",
    "heavy shower snow": "heavy_snow",

    # Atmosphere
    "mist": "fog",
    "smoke": "smoke",
    "haze": "haze",
    "sand/dust whirls": "dust_storm",
    "fog": "fog",
    "sand": "dust_storm",
    "dust": "dust_storm",
    "volcanic ash": "ash",
    "squalls": "squalls",
    "tornado": "tornado",
}

# Mood modifiers based on weather
WEATHER_MOOD_MODIFIERS = {
    "clear": 0,
    "partly_cloudy": 0,
    "cloudy": 0.5,
    "overcast": 1,
    "drizzle": 0.5,
    "light_rain": 1,
    "rain": 1.5,
    "heavy_rain": 2,
    "storm": 3,
    "thunderstorm": 3,
    "severe_thunderstorm": 4,
    "fog": 2,
    "light_snow": 0.5,
    "snow": 1,
    "heavy_snow": 2,
    "sleet": 1.5,
    "freezing_rain": 2,
    "haze": 1,
    "smoke": 2,
    "dust_storm": 3,
    "tornado": 5,
}


@dataclass
class WeatherData:
    """Current weather data."""
    condition: str           # Game weather condition
    description: str         # Human-readable description
    temperature_f: float     # Fahrenheit
    temperature_c: float     # Celsius
    feels_like_f: float
    feels_like_c: float
    humidity: int            # Percentage
    wind_speed: float        # mph
    wind_direction: str      # "N", "NE", "E", etc.
    visibility: float        # miles
    clouds: int              # Percentage cloud cover

    # For narrative
    mood_modifier: float     # How much this affects world mood
    narrative_hint: str      # Suggestion for DM

    # Metadata
    location: str
    timestamp: str
    raw_condition: str       # Original from API

    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "description": self.description,
            "temperature_f": self.temperature_f,
            "temperature_c": self.temperature_c,
            "feels_like_f": self.feels_like_f,
            "feels_like_c": self.feels_like_c,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "visibility": self.visibility,
            "clouds": self.clouds,
            "mood_modifier": self.mood_modifier,
            "narrative_hint": self.narrative_hint,
            "location": self.location,
            "timestamp": self.timestamp,
            "raw_condition": self.raw_condition,
        }

    def get_narrative_description(self) -> str:
        """Get a narrative description for the DM."""
        parts = []

        # Temperature
        if self.temperature_f < 32:
            parts.append("The cold bites through every layer")
        elif self.temperature_f < 50:
            parts.append("A chill hangs in the air")
        elif self.temperature_f > 90:
            parts.append("The heat is oppressive")
        elif self.temperature_f > 75:
            parts.append("It's warm")

        # Main condition
        condition_descriptions = {
            "clear": "The sky is clear",
            "partly_cloudy": "Clouds drift lazily overhead",
            "cloudy": "Gray clouds blanket the sky",
            "overcast": "Heavy clouds press down on the world",
            "fog": "Fog obscures everything beyond arm's reach",
            "drizzle": "A fine mist coats everything",
            "light_rain": "Light rain patters against surfaces",
            "rain": "Rain falls steadily",
            "heavy_rain": "Rain hammers down relentlessly",
            "storm": "The storm rages",
            "thunderstorm": "Thunder rolls and lightning cracks the sky",
            "severe_thunderstorm": "The storm is apocalyptic in its fury",
            "light_snow": "Snowflakes drift down gently",
            "snow": "Snow falls in earnest",
            "heavy_snow": "A blizzard howls",
        }

        if self.condition in condition_descriptions:
            parts.append(condition_descriptions[self.condition])

        # Wind
        if self.wind_speed > 30:
            parts.append("Wind tears at anything not anchored down")
        elif self.wind_speed > 15:
            parts.append("A strong wind pushes against you")

        # Visibility
        if self.visibility < 0.5:
            parts.append("You can barely see your hand in front of your face")
        elif self.visibility < 2:
            parts.append("Visibility is poor")

        return ". ".join(parts) + "." if parts else f"The weather is {self.condition}."


class WeatherSync:
    """
    Syncs real-world weather to game weather.

    Uses OpenWeatherMap API with caching to stay within free tier limits.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.cache_file = self.data_dir / "weather_cache.json"
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        self.cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(minutes=30)
        self._load_cache()

    def _load_cache(self):
        """Load cached weather data."""
        if self.cache_file.exists():
            try:
                self.cache = json.loads(self.cache_file.read_text())
            except:
                self.cache = {}

    def _save_cache(self):
        """Save weather cache."""
        try:
            self.cache_file.write_text(json.dumps(self.cache, indent=2))
        except Exception as e:
            print(f"[Weather] Cache save failed: {e}")

    def _is_cache_valid(self, location: str) -> bool:
        """Check if cached data is still fresh."""
        if location not in self.cache:
            return False
        cached = self.cache[location]
        cached_time = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
        return datetime.now() - cached_time < self.cache_duration

    def _wind_direction(self, degrees: float) -> str:
        """Convert wind degrees to compass direction."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(degrees / 22.5) % 16
        return directions[idx]

    def _generate_narrative_hint(self, condition: str, temp_f: float) -> str:
        """Generate a narrative hint for the DM based on weather."""
        hints = []

        if condition in ["fog", "heavy_rain", "storm", "heavy_snow"]:
            hints.append("Movement is difficult. Visibility is poor.")

        if condition in ["thunderstorm", "severe_thunderstorm"]:
            hints.append("The noise could mask other sounds - or attract attention.")

        if temp_f < 32:
            hints.append("Characters without shelter risk hypothermia.")
        elif temp_f > 95:
            hints.append("Heat exhaustion is a real danger without water.")

        if condition == "clear" and temp_f > 60 and temp_f < 80:
            hints.append("Good weather for travel or outdoor activities.")

        if not hints:
            return "Weather is manageable."

        return " ".join(hints)

    def get_weather(self, location: str) -> Optional[WeatherData]:
        """
        Get weather for a location.

        location can be:
        - City name: "Denver, CO"
        - Zip code: "80202"
        - Coordinates: "39.7392,-104.9903"
        """
        if not self.api_key:
            # Return default weather if no API key
            return WeatherData(
                condition="clear",
                description="Clear sky (no API key configured)",
                temperature_f=70,
                temperature_c=21,
                feels_like_f=70,
                feels_like_c=21,
                humidity=50,
                wind_speed=5,
                wind_direction="N",
                visibility=10,
                clouds=0,
                mood_modifier=0,
                narrative_hint="Configure OPENWEATHERMAP_API_KEY for real weather sync.",
                location=location,
                timestamp=datetime.now().isoformat(),
                raw_condition="clear",
            )

        # Check cache first
        if self._is_cache_valid(location):
            cached = self.cache[location]
            return WeatherData(**cached["data"])

        # Fetch from API
        try:
            # Determine query type
            if "," in location and all(c in "0123456789.,-" for c in location.replace(" ", "")):
                # Coordinates
                lat, lon = location.split(",")
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat.strip()}&lon={lon.strip()}&appid={self.api_key}&units=imperial"
            elif location.isdigit() or (len(location) == 5 and location.replace("-", "").isdigit()):
                # Zip code
                url = f"https://api.openweathermap.org/data/2.5/weather?zip={location},US&appid={self.api_key}&units=imperial"
            else:
                # City name
                url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.api_key}&units=imperial"

            response = requests.get(url, timeout=10)
            if not response.ok:
                print(f"[Weather] API error: {response.status_code}")
                return None

            data = response.json()

            # Parse response
            raw_condition = data["weather"][0]["description"].lower()
            condition = WEATHER_MAP.get(raw_condition, "clear")

            weather = WeatherData(
                condition=condition,
                description=data["weather"][0]["description"],
                temperature_f=data["main"]["temp"],
                temperature_c=(data["main"]["temp"] - 32) * 5/9,
                feels_like_f=data["main"]["feels_like"],
                feels_like_c=(data["main"]["feels_like"] - 32) * 5/9,
                humidity=data["main"]["humidity"],
                wind_speed=data.get("wind", {}).get("speed", 0),
                wind_direction=self._wind_direction(data.get("wind", {}).get("deg", 0)),
                visibility=data.get("visibility", 10000) / 1609.34,  # meters to miles
                clouds=data.get("clouds", {}).get("all", 0),
                mood_modifier=WEATHER_MOOD_MODIFIERS.get(condition, 0),
                narrative_hint=self._generate_narrative_hint(condition, data["main"]["temp"]),
                location=f"{data.get('name', location)}, {data.get('sys', {}).get('country', '')}",
                timestamp=datetime.now().isoformat(),
                raw_condition=raw_condition,
            )

            # Cache it
            self.cache[location] = {
                "fetched_at": datetime.now().isoformat(),
                "data": weather.to_dict(),
            }
            self._save_cache()

            return weather

        except Exception as e:
            print(f"[Weather] Fetch failed: {e}")
            return None

    def sync_to_world_state(self, world_state, location: str) -> bool:
        """
        Sync weather to a WorldState object.

        Returns True if weather was updated.
        """
        weather = self.get_weather(location)
        if not weather:
            return False

        world_state.weather = weather.condition

        # Optionally affect mood based on weather
        # (gentle nudge, not override)

        return True


# Global instance
_weather_sync: Optional[WeatherSync] = None


def get_weather_sync() -> WeatherSync:
    """Get the global weather sync instance."""
    global _weather_sync
    if _weather_sync is None:
        _weather_sync = WeatherSync()
    return _weather_sync


def init_weather_sync(data_dir: Optional[Path] = None) -> WeatherSync:
    """Initialize the weather sync system."""
    global _weather_sync
    _weather_sync = WeatherSync(data_dir)
    return _weather_sync
