mkdir -p ./endurance/
mkdir -p ./endurance/postgres/
docker-compose -f docker-compose.yml up -d --remove-orphans

# Connect to sql with:
# psql -U postgres -p 8432 -h localhost
# Requires password (see docker-compose.yml)
