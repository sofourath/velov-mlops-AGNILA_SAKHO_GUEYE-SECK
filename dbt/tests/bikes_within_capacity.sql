{{ config(
    severity='error',
    warn_if='>0',
    error_if='>10'
) }}

select
    station_id,
    timestamp,
    bikes_available,
    capacity
from {{ ref('bronze_velov') }}
where bikes_available < 0
   or bikes_available > capacity