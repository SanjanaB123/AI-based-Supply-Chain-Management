#!/bin/bash
set -e

# What this script does:
# 1. If this container is the "init" job → migrate DB + create admin user
# 2. If this container is the webserver → wait for DB, then start
# 3. If this container is scheduler/dag-processor/triggerer → wait for DB, start with dummy HTTP server

echo "Starting Airflow component: $@"

# Function: wait until the database is migrated and ready
wait_for_db() {
  echo "Waiting for database to be ready..."
  for i in $(seq 1 60); do
    if airflow db check 2>/dev/null; then
      echo "Database is ready!"
      return 0
    fi
    echo "Database not ready yet (attempt $i/60)... waiting 5s"
    sleep 5
  done
  echo "ERROR: Database not ready after 5 minutes"
  return 1
}

# Check what command was passed
COMMAND="${1:-}"

case "$COMMAND" in
  "db-init")
    # Init job: migrate DB and create admin user
    echo "Running database migration..."
    airflow db migrate
    echo "Creating admin user..."
    airflow users create \
      --username admin \
      --password admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com || true
    echo "Database initialization complete!"
    ;;

  "api-server"|"webserver")
    # Webserver: wait for DB, then start
    wait_for_db
    shift
    exec airflow api-server "$@"
    ;;

  "scheduler")
    # Scheduler: wait for DB, start scheduler + dummy HTTP server
    wait_for_db
    python -m http.server 8080 --bind 0.0.0.0 &
    exec airflow scheduler
    ;;

  "dag-processor")
    # DAG Processor: wait for DB, start processor + dummy HTTP server
    wait_for_db
    python -m http.server 8080 --bind 0.0.0.0 &
    exec airflow dag-processor
    ;;

  "triggerer")
    # Triggerer: wait for DB, start triggerer + dummy HTTP server
    wait_for_db
    python -m http.server 8080 --bind 0.0.0.0 &
    exec airflow triggerer
    ;;

  *)
    # Default: just run whatever command was passed
    exec "$@"
    ;;
esac
