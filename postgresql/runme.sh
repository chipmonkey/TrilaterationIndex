export PGPASSWORD=mysecretpassword
mkdir -p ./endurance/data
mkdir -p ./endurance/input
mkdir -p ./endurance/output
cp ../data/lat_long_categorized.csv ./endurance/input
docker-compose -f postgres.yml up -d --remove-orphans

# to connect use:
# psql -h localhost -p 5432 -U postgres
# or as chipmonkey or whatever

sleep 10  # Hack so that psql is up by the time we run commands

psql -h localhost -p 5432 -U postgres -e -f setup_database.sql

# psql -h localhost -p 5432 -U postgres -e -c 'select * from v_results;' > query_timings.out

psql -h localhost -p 5432 -U postgres -e -c 'select * from v_results;' -f query_timings.csv -A -F','

