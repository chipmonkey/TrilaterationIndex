export PGPASSWORD=mysecretpassword
mkdir -p ./endurance/data
mkdir -p ./endurance/input
cp ../data/lat_long_categorized.csv ./endurance/input
docker-compose -f postgres.yml up -d --remove-orphans

# to connect use:
# psql -h localhost -p 5432 -U postgres
# or as chipmonkey or whatever

psql -h localhost -p 5432 -U postgres -e -f setup_database.sql

psql -h localhost -p 5432 -U postgres -e -c 'select * from query_timings;'
