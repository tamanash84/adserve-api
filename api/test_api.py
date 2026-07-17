from govt_demography_api import get_age_group, get_country_origin
from overpass_poi_api import get_poi_features
from weather_7timer_api import get_weather_features
from ticketmaster_event_api import get_event_features
from datetime import datetime, timezone

lat = 52.3120164
lon = 4.9436886
pc = 1106
year = 2025

# res1 = get_age_group(pc, year)
# res2 = get_country_origin(pc, year)

# res3 = get_poi_features(
#         lat=lat,
#         lon=lon,
#         bands=((0, 50), (50, 200), (200, 800)),
#         missing_value=-1.0,
#         add_any=True,
#         add_density=False,
#         add_open_now=True,
#         tz="Europe/Amsterdam",
#         self_exclude_keywords=("albert heijn", "ah"),
#     )

# res4 = get_weather_features(lat=lat, 
#                             lon=lon, 
#                             now_utc=datetime.now(timezone.utc),
#                             tz_name="Europe/Amsterdam",
#                             time_windows_hours=((0, 3),),
#                             include_onehots=True)

res5 = get_event_features(now_utc=datetime.now(timezone.utc),
                          lat=lat, 
                          lon=lon)
