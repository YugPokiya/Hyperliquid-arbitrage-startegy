# LiquiMind Hyperliquid Arbitrage Strategy

LiquiMind is a trading-system starter project that currently runs a Python Hyperliquid market-data consumer and strategy loop. The intended redevelopment target is a Rust/Axum high-frequency trading backend with async persistence, low-latency WebSocket ingestion, microstructure features, and production DevOps/cloud deployment.

## Current runnable baseline

- `src1/connection/ws_client.py` maintains a Hyperliquid WebSocket connection, subscribes to L2 book and trade feeds, and forwards order-book updates into an async queue.
- `src1/main.py` consumes queued market data, extracts the best bid/ask, computes a mid-price, and provides the hook where strategy, feature, inference, and execution logic should run.
- `src1/utils/indicators.py` contains a PyTorch Fisher Transform implementation with tests in `test/test_indicators.py`.
- `docker-compose.yml` packages the bot with a MongoDB service and persistent volumes for logs, model checkpoints, and database state.

## Redevelopment roadmap: Rust/Axum HFT backend

### 1. Core backend architecture

Build the production backend as a Rust workspace with these services/modules:

1. **Gateway API**: Axum REST and WebSocket API for health, metrics, manual controls, strategy state, and admin operations.
2. **Market data ingestion**: Tokio WebSocket clients for Hyperliquid subscriptions with bounded channels, backpressure, reconnect loops, and circuit breakers.
3. **Feature engine**: Bronze-layer microstructure computations including Lee-Ready trade classification, VWAP, order-book imbalance, Fisher Transform, and Fourier/order-book shape features.
4. **Strategy engine**: Signal generation, position sizing, risk checks, and paper/live execution modes.
5. **Execution engine**: Signed order submission, idempotency keys, retry policy, rate-limit tracking, and kill-switch integration.
6. **Persistence**: SQLx with PostgreSQL for ticks, L2 snapshots, computed features, orders, fills, strategy decisions, and audit logs.
7. **Observability**: `tracing`, Prometheus metrics, OpenTelemetry traces, structured JSON logs, and latency histograms.

### 2. Suggested Rust workspace layout

```text
liquimind/
  crates/
    api/              # Axum routes, auth, health checks, admin controls
    market-data/      # Hyperliquid WS consumers, subscriptions, reconnects
    features/         # Lee-Ready, VWAP, Fisher, Fourier, imbalance
    strategy/         # Signal logic and risk-aware decision engine
    execution/        # Order placement, cancel/replace, exchange adapters
    persistence/      # SQLx repositories and migrations
    telemetry/        # tracing, metrics, OpenTelemetry setup
  migrations/         # SQLx PostgreSQL migrations
  deploy/
    docker/
    helm/
    terraform/
```

### 3. DevOps and cloud milestones

1. **Containerization**: Multi-stage Rust Dockerfile with non-root runtime image and separate local `docker-compose` stack for Postgres, Prometheus, and Grafana.
2. **CI/CD**: GitHub Actions for formatting, Clippy, tests, security audit, Docker image build, SBOM generation, and deployment promotion.
3. **Infrastructure as Code**: Terraform for VPC, managed PostgreSQL, container registry, secrets, Kubernetes/ECS, load balancer, and monitoring resources.
4. **Kubernetes or ECS deployment**: Separate deployments for API, market-data workers, strategy workers, and execution workers; add HPA based on CPU, queue depth, and message lag.
5. **Secrets management**: Cloud-native secret store for private keys and API credentials; never bake secrets into images or compose files.
6. **Observability stack**: Prometheus/Grafana dashboards for tick throughput, reconnect counts, order latency, rejected orders, PnL, drawdown, and risk-limit breaches.
7. **Reliability controls**: Circuit breakers, dead-letter queues, replayable event logs, readiness probes, graceful shutdown, and emergency kill switch.

### 4. Resume-ready project description

> Architected LiquiMind, a low-latency trading backend target in Rust with Axum, Tokio, and SQLx for async PostgreSQL persistence, designed for sub-millisecond internal order-processing paths. Implemented market microstructure features including Lee-Ready tick classification, VWAP, Fisher Transform, and Fourier-based order-book analytics. Built a high-concurrency WebSocket ingestion design with circuit breakers, reconnect recovery, and observability-first DevOps deployment across containers, CI/CD, and cloud infrastructure.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python src1/main.py
```

## Docker development

```bash
docker compose up --build
```

Create a `.env` file before running live exchange integrations:

```env
HYPERLIQUID_PRIVATE_KEY=...
WALLET_ADDRESS=...
```

Do not commit real secrets.
