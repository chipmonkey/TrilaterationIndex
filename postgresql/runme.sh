docker-compose -f postgres.yml up -d --remove-orphans

# to connect use:
# psql -h localhost -p 5432 -U postgres
