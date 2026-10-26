select *
from read_csv_auto(
    '{{ var("velov_bronze_csv") }}',
    header = true
)
