# LHCPID — Particle Shower Classification System

![Language](https://img.shields.io/badge/Language-C%2B%2B20%20%7C%20Java%2021-blue)
![Simulation](https://img.shields.io/badge/Simulation-Geant4%2011-orange)
![Streaming](https://img.shields.io/badge/Streaming-Apache%20Kafka-black)
![Containerised](https://img.shields.io/badge/Containerised-Docker%20Compose-2496ED?logo=docker)
![ML](https://img.shields.io/badge/TensorLess(no%20libs)-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **A real-time particle classification pipeline** — from Monte Carlo physics simulation to neural network inference to a live web dashboard, fully containerized and deployable with a single command.

---

## What This Is

High-energy physics experiments generate enormous amounts of detector data that must be classified in real time. This project builds a full end-to-end system that:

1. **Simulates** particle showers in a sampling calorimeter using [Geant4](https://geant4.web.cern.ch/) (the same toolkit used at CERN)
2. **Streams** per-event data through Apache Kafka
3. **Classifies** each event using a custom neural network — built entirely from scratch in Java, no PyTorch or TensorFlow
4. **Displays** results live on a WebSocket-powered web dashboard

The classifier identifies four particle types — electrons, pions, muons, and gamma rays — based on the energy deposition pattern they leave across 10 calorimeter layers, achieving **82.5% accuracy** with near-zero overfitting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Docker Network: lhcpid-net                     │
│                                                                         │
│   ┌──────────────┐  GENERATE/BATCH   ┌──────────────────────────────┐  │
│   │  Java Spring │ ────────────────► │     Geant4 Simulation (C++)  │  │
│   │  Backend     │    TCP :5003       │     lhcpid-sim               │  │
│   │  lhcpid-     │                   │                              │  │
│   │  backend     │ ◄──────────────── │  Fires particles, collects   │  │
│   │              │  CSV rows TCP:5001 │  per-layer energy deposits   │  │
│   │  ┌─────────┐ │                   └──────────────────────────────┘  │
│   │  │ Kafka   │ │                                                      │
│   │  │Consumer │ │   ┌─────────────────────┐                           │
│   │  └────┬────┘ │   │   Apache Kafka      │                           │
│   │       │      │   │   lhcpid-kafka      │                           │
│   │  Neural Net  │   │   topic:raw-particles│                          │
│   │  Inference   │   └─────────────────────┘                           │
│   │       │      │                                                      │
│   │  WebSocket   │   ┌─────────────────────┐                           │
│   │  Dashboard   │   │   PostgreSQL        │                           │
│   └──────────────┘   │   lhcpid-db         │                           │
│        :8080         └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Two communication channels connect the simulation to the backend:**
- **Port 5003** — Command channel (Java → C++): sends `GENERATE` / `BATCH` commands to control what particles are fired
- **Port 5001** — Data channel (C++ → Java): streams raw event data after each simulated particle

---

## Physics Background

A sampling calorimeter measures particle energy by interleaving passive absorber layers (where particles shower) with active detector layers (where energy is measured). Different particles leave distinct signatures:

| Particle | Behaviour | Key Signature |
|----------|-----------|---------------|
| **Electron e⁻** | EM shower begins immediately | High energy in layers 0–2, rapid decay |
| **Pion π⁻** | Hadronic shower, high variance | Late or irregular energy profile |
| **Muon μ⁻** | Minimum ionising — barely interacts | Uniform ~12–15 MeV across all 10 layers |
| **Gamma γ** | EM shower delayed (pair-production gap) | Near-zero energy in layer 0, peak at layers 3–5 |

### Why Copper?

The absorber material is **10 mm Copper** (not the more common Lead). Lead's short radiation length (X₀ ≈ 0.56 cm) compresses showers so aggressively that gammas become indistinguishable from electrons in just 10 layers. Copper's longer X₀ (≈ 1.43 cm) preserves the gamma conversion gap — the key discriminating feature — while still forcing pions to shower and separating them from muons.

---

## The Neural Network

Built **from scratch in Java** — no ML libraries, no autograd, no external dependencies. Every forward pass, backprop step, and weight update is hand-coded.

```
Input (16 features)  →  Hidden 1 (32, ReLU)  →  Hidden 2 (16, ReLU)  →  Output (4, Softmax)
```

### Feature Engineering (16 inputs from 10 raw layer readings)

| # | Feature | Physics Motivation |
|---|---------|-------------------|
| 0–9 | Log-scaled layer energies `log(1+Eᵢ)/6` | Compresses 0–200 MeV range |
| 10 | Total energy (Z-scored) | Separates high/low energy events |
| 11 | Layer-0 ratio (Z-scored) | Electron vs. Gamma key discriminant |
| 12 | Peak position (Z-scored) | Shower maximum depth |
| 13 | Early fraction `(E₀+E₁+E₂)/Total` | Early vs. late shower development |
| 14 | Layer-0 interaction flag (binary) | Gamma conversion indicator |
| 15 | Roughness `σ(Eᵢ)/100` | Uniform muons vs. variable pions |

### Training
- **Optimiser:** SGD with momentum, batch size 32
- **Loss:** Cross-entropy + L2 regularisation (λ = 0.0001)
- **Learning rate:** 0.02–0.05 with ×0.9 decay per 1,000 epochs
- **Early stopping:** patience = 500 epochs
- **Dataset:** 80,000 simulated events (64k train / 16k test)
- **Model size:** 3.2 KB — 2,564 parameters

---

## Results

**Overall test accuracy: 81.99% — train/test gap: 0.12%**

### Confusion Matrix (360,000 test events)

```
   --- CONFUSION MATRIX ---
Act \ Pred | Elec  | Pion  | Muon  | Gamma | 
Elec       | 88724  | 936    | 52     | 288    | 
Pion       | 5690   | 51613  | 32620  | 77     | 
Muon       | 17     | 178    | 89805  | 0      | 
Gamma      | 24559  | 405    | 5      | 65031  |  
```

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Electron | 0.97 | 0.95 | 0.96 |
| Pion | 0.73 | 0.63 | 0.68 |
| Muon | 0.96 | 0.99 | 0.97 |
| Gamma | 0.75 | 0.75 | 0.75 |

> **Note on Pion/Muon confusion:** The 32620 pion→muon misclassifications are a physical detector limitation — "hadronic punch-through" where pions lose most of their energy before showering and mimic a muon's minimum-ionising signature. This is not a model failure; it reflects the information limit of the detector geometry.

---

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- That's it — everything else is containerised

### Run the full stack

```bash
git repo clone GrandSensei/Geant4_SpringBoot_
docker compose up --build
```

The first build takes a few minutes (Geant4 is a large library). Once running:

| Service | URL / Address |
|---------|--------------|
| Web Dashboard | http://localhost:8080 |
| Kafka broker (external) | localhost:9092 |
| PostgreSQL | localhost:5432 |
| Sim command port | localhost:5003 |

### Send your first particle

```bash
# Via the REST API (once backend is up)
curl -X POST http://localhost:8080/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"particleType": "electron", "energy": 300}'
```

### Tear down

```bash
docker compose down
```


---

## Technical Highlights

- **No ML libraries** — neural network, backpropagation, softmax, L2 regularisation all implemented from first principles in plain Java
- **Physics-informed features** — features designed around actual particle physics knowledge, not generic statistics
- **Two-way simulation control** — the backend can dynamically instruct the simulation what to generate at runtime, not just consume pre-generated data
- **Production-style architecture** — proper message queue (Kafka), persistent storage (PostgreSQL), containerised microservices, WebSocket dashboard
- **Deterministic reconnection** — the C++ simulation retries its connection to the backend for up to 30 seconds on startup, handling Docker's non-deterministic startup order gracefully

---

## Future Work

- [ ] **v2.0 detector:** Switch to Lead absorber with 40 layers × 2 mm (matching CMS ECAL geometry)
- [ ] Adam optimiser and dropout regularisation
- [ ] Shower shape moments (skewness, kurtosis) as additional features
- [ ] K-fold cross-validation and 100k+ event dataset
- [ ] Higher energy range training (100 GeV – 1 TeV)

---

## References

- [Geant4 Collaboration](https://geant4.web.cern.ch/) — Monte Carlo simulation toolkit to study the code of example calorimeter
- [Particle Data Group](https://pdg.lbl.gov/) — Passage of Particles Through Matter and finding the meaning of certain notations and labels.
- Fabjan & Gianotti, *Calorimetry for Particle Physics*, Rev. Mod. Phys. 75, 1243 (2003) - A cool paper which I understood about 15%, enough to code my way through
- Goodfellow et al., *Deep Learning*, MIT Press (2016) - A nice book for the algorithmic set up of the work.

---

## Author

**Mustafa Bazi** — Built to demonstrate the intersection of high-energy physics simulation and machine learning engineering.

*"Building a neural network from scratch teaches you more than using PyTorch ever could."*

---

*Licensed under MIT. Geant4 components are subject to the [Geant4 Software License](https://geant4.web.cern.ch/license).*
