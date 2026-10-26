select
    cast(station_id as integer) as station_id,
    cast(timestamp as timestamp with time zone) as timestamp,
    cast(capacity as integer) as capacity,
    cast(bikes_available as integer) as bikes_available,
    cast(temperature as double) as temperature,
    cast(is_raining as boolean) as is_raining
from {{ ref('bronze_velov') }}
where station_id is not null
  and timestamp is not null
  and capacity > 0
  and bikes_available between 0 and capacity
  and temperature is not null
  and is_raining is not null
