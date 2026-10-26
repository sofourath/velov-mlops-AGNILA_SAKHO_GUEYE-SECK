select
    cast(station_id as integer) as station_id,
    cast(timestamp as timestamp with time zone) as timestamp,
    cast(capacity as integer) as capacity,
    cast(bikes_available as integer) as bikes_available,
    cast(temperature as double) as temperature,
    cast(is_raining as boolean) as is_raining
from {{ ref('velov_silver') }}
