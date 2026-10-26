{{ config(
    materialized="table",
    contract={"enforced": true}
) }}

select
    station_id,
    timestamp,
    capacity,
    bikes_available,
    temperature,
    is_raining
from {{ ref("velov_observations") }}
where station_id >= 1
  and capacity > 0
  and capacity <= 100
  and bikes_available >= 0
  and bikes_available <= capacity
  and temperature between -30 and 50
