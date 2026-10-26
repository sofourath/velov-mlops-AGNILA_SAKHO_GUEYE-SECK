{{ config(materialized="table") }}

with typed as (
    select
        cast(station_id as integer) as station_id,
        cast(timestamp as timestamptz) as timestamp,
        cast(capacity as integer) as capacity,
        cast(bikes_available as integer) as bikes_available,
        cast(temperature as double) as temperature,
        cast(is_raining as boolean) as is_raining
    from {{ ref("bronze_velov") }}
),
ranked as (
    select
        *,
        row_number() over (
            partition by station_id, timestamp
            order by timestamp desc
        ) as row_number
    from typed
)
select
    station_id,
    timestamp,
    capacity,
    bikes_available,
    temperature,
    is_raining
from ranked
where row_number = 1
