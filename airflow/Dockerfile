# Builds a custom docker image thas Airflow + DAGs + Python dependencies
# This image will be used by Cloud Run to run Airflow in a serverless environment
# Custom Airflow image for Cloud Run deployment
# Bakes in DAGs, scripts, params, and all dependencies
FROM apache/airflow:3.0.3-python3.12

# Install git (needed by DVC)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
USER airflow

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake in DAGs, scripts, params, and DVC config
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root params.yaml /opt/airflow/params.yaml
COPY --chown=airflow:root .dvc/config /opt/airflow/.dvc/config

# Entrypoint script that handles DB migration before starting
COPY --chown=airflow:root entrypoint.sh /opt/airflow/entrypoint.sh
RUN chmod +x /opt/airflow/entrypoint.sh

# Set environment
ENV PYTHONPATH=/opt/airflow \
    PARAMS_PATH=/opt/airflow/params.yaml

ENTRYPOINT ["/opt/airflow/entrypoint.sh"]
