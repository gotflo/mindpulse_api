# 🧠 Cognitive State API

**Backend Python pour la mesure, l'analyse et la prédiction en temps réel de l'état cognitif à partir du PPG (Polar Verity Sense).**

Ce serveur constitue le cœur du système : il acquiert les données physiologiques via BLE, les traite en temps réel, prédit les états cognitifs (stress, charge cognitive, fatigue mentale) et diffuse les résultats vers l'application Flutter via WebSocket.

---

## 📋 Table des matières

- [Architecture globale](#-architecture-globale)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Lancement](#-lancement)
- [Pipeline temps réel](#-pipeline-temps-réel)
- [Modules détaillés](#-modules-détaillés)
  - [Acquisition BLE](#1-acquisition-ble)
  - [Traitement du signal](#2-traitement-du-signal)
  - [Extraction de features HRV](#3-extraction-de-features-hrv)
  - [Modèle IA](#4-modèle-ia)
  - [Stockage](#5-stockage)
  - [Analyse & recommandations](#6-analyse--recommandations)
- [API REST](#-api-rest)
- [WebSocket (temps réel)](#-websocket-temps-réel)
- [Base de données](#-base-de-données)
- [Configuration](#-configuration)
- [Mode heuristique vs modèle entraîné](#-mode-heuristique-vs-modèle-entraîné)

---

## 🏗 Architecture globale

```
┌──────────────────────┐
│  Polar Verity Sense  │
│       (PPG/BLE)      │
└──────────┬───────────┘
           │ BLE (HR + PPI)
           ▼
┌──────────────────────────────────────────────────┐
│              Backend Python (ce serveur)          │
│                                                   │
│  Acquisition → Nettoyage → Features → IA → Stock │
│                                                   │
│  REST API  +  WebSocket (Socket.IO)              │
└──────────┬───────────────────────┬───────────────┘
           │ WebSocket             │ REST
           ▼                       ▼
┌──────────────────────┐  ┌────────────────┐
│    Flutter App       │  │  Outil desktop │
│  (affichage temps    │  │  (analyse CSV, │
│   réel, sessions)    │  │  réentraîne-   │
└──────────────────────┘  │  ment modèle)  │
                          └────────────────┘
```

---

## 📂 Structure du projet

```
api/
├── run.py                              # Point d'entrée du serveur
├── requirements.txt                    # Dépendances Python
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── factory.py                      # App factory (composition & injection)
│   │
│   ├── config/
│   │   └── settings.py                 # Dataclasses de configuration
│   │
│   ├── acquisition/
│   │   └── polar_client.py             # Client BLE Polar Verity Sense
│   │
│   ├── signal/
│   │   ├── ppi_cleaning.py             # Nettoyage PPI / suppression artéfacts
│   │   └── windowing.py                # Fenêtres glissantes temps réel
│   │
│   ├── features/
│   │   └── hrv_features.py             # Extraction features HRV (14 métriques)
│   │
│   ├── ml/
│   │   ├── model.py                    # Modèle cognitif (sklearn ou heuristique)
│   │   ├── inference.py                # Pipeline d'inférence + lissage + tendance
│   │   └── models/                     # Dossier pour les fichiers .joblib
│   │
│   ├── storage/
│   │   ├── database.py                 # Couche SQLite (3 tables + index)
│   │   └── session_manager.py          # Gestion des sessions + export
│   │
│   ├── domain/
│   │   ├── pipeline.py                 # Pipeline temps réel (orchestration)
│   │   └── analysis_service.py         # Analyse historique & recommandations
│   │
│   └── api/
│       ├── routes.py                   # Endpoints REST
│       └── socket_events.py            # Événements WebSocket (auto-monitoring)
│
├── data/
│   ├── cognitive.db                    # Base SQLite (auto-créée)
│   ├── sessions/                       # Données de sessions brutes
│   └── exports/                        # Fichiers CSV exportés
│
└── tests/
```

---

## 🚀 Installation

### Prérequis
- Python 3.11+
- Polar Verity Sense (firmware récent)
- Bluetooth Low Energy activé sur la machine

### Étapes

```bash
# 1. Cloner le projet
cd D:/projet_mobile/cognitive/api

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

| Package | Rôle |
|---------|------|
| `flask` | Framework web |
| `flask-socketio` | WebSocket temps réel |
| `flask-cors` | Cross-origin (Flutter) |
| `bleak` | Communication BLE |
| `numpy` | Calcul numérique |
| `scipy` | Traitement du signal (Welch PSD, interpolation) |
| `scikit-learn` | Modèle ML (si entraîné) |
| `joblib` | Sérialisation modèle |

---

## ▶ Lancement

```bash
# Lancement par défaut (0.0.0.0:5000)
python run.py

# Avec options
python run.py --host 127.0.0.1 --port 8080 --debug

# Via variables d'environnement
HOST=0.0.0.0 PORT=5000 DEBUG=true python run.py
```

Le serveur expose :
- **REST API** sur `http://localhost:5000/api/`
- **WebSocket** sur `http://localhost:5000` (Socket.IO)

---

## ⚡ Pipeline temps réel

Le cœur du système est un pipeline en 6 étapes, exécuté toutes les **5 secondes** sur une fenêtre glissante de **30 secondes** :

```
Polar Verity Sense (BLE)
    │
    │  HR + PPI bruts
    ▼
┌─────────────────────────────────────────┐
│  1. NETTOYAGE PPI                       │
│     • Filtre physiologique [300–2000ms] │
│     • Détection ectopiques (Δ > 20%)    │
│     • Interpolation cubique             │
│     • Score de qualité du segment       │
├─────────────────────────────────────────┤
│  2. FENÊTRAGE                           │
│     • Buffer glissant 30s               │
│     • Émission toutes les 5s            │
│     • Seuil 80% de remplissage          │
├─────────────────────────────────────────┤
│  3. EXTRACTION FEATURES HRV             │
│     • 7 features temporelles            │
│     • 4 features fréquentielles (Welch) │
│     • 3 features non-linéaires          │
│     = 14 features au total              │
├─────────────────────────────────────────┤
│  4. PRÉDICTION IA                       │
│     • Modèle sklearn OU heuristique     │
│     • 3 scores : stress, charge, fatigue│
│     • Scores continus 0–100             │
├─────────────────────────────────────────┤
│  5. LISSAGE & TENDANCE                  │
│     • EMA (α=0.3) sur les scores        │
│     • Régression linéaire fatigue       │
│     • Prédiction fatigue à +10 min      │
│     • Indice de confiance (R²)          │
├─────────────────────────────────────────┤
│  6. DIFFUSION & STOCKAGE                │
│     • WebSocket → Flutter (temps réel)  │
│     • SQLite → data_points (si session) │
└─────────────────────────────────────────┘
```

---

## 📦 Modules détaillés

### 1. Acquisition BLE

**Fichier** : `app/acquisition/polar_client.py`

Gère la connexion Bluetooth avec le Polar Verity Sense via la bibliothèque `bleak`.

| Fonctionnalité | Détail |
|----------------|--------|
| Scan BLE | Recherche du device par nom ("Polar Verity Sense") |
| Connexion | Avec reconnexion automatique (3 tentatives, délai 2s) |
| Streaming HR | UUID `00002a37-...` — parsing flags 8/16 bits |
| Streaming PPI | UUID `fb005c81-...` — format 6 octets (HR + PPI + error + flags) |
| Batterie | UUID `00002a19-...` — lecture niveau batterie |
| Qualité signal | Suivi continu du flag skin_contact (fenêtre de 50 samples) |

**États de connexion** : `DISCONNECTED → SCANNING → CONNECTING → CONNECTED → STREAMING → ERROR`

---

### 2. Traitement du signal

**Fichiers** : `app/signal/ppi_cleaning.py`, `app/signal/windowing.py`

#### Nettoyage PPI
```
Entrée : PPI bruts du capteur
  ↓
Filtre physiologique : 300ms ≤ PPI ≤ 2000ms
  ↓
Détection ectopiques : |ΔRR| / RR > 20% → retiré
  ↓
Interpolation cubique des samples retirés
  ↓
Sortie : RR nettoyés + ratio de qualité
```

- **Qualité minimum** : 80% de samples valides, sinon avertissement
- **Ratio qualité** : `nb_valides / nb_total`

#### Fenêtrage glissant
- **Taille** : 30 secondes (configurable)
- **Pas** : 5 secondes (configurable)
- **Seuil d'émission** : buffer rempli à ≥ 80%
- Reconstruction des timestamps depuis les durées PPI

---

### 3. Extraction de features HRV

**Fichier** : `app/features/hrv_features.py`

**14 features extraites par fenêtre :**

#### Domaine temporel (7 features)

| Feature | Formule | Signification |
|---------|---------|---------------|
| `mean_hr` | 60000 / mean(RR) | Fréquence cardiaque moyenne |
| `mean_rr` | mean(RR) | Intervalle RR moyen (ms) |
| `sdnn` | std(RR) | Variabilité globale |
| `rmssd` | √mean(ΔRR²) | Variabilité court-terme (parasympathique) |
| `pnn50` | %(\|ΔRR\| > 50ms) | Activité parasympathique |
| `sdsd` | std(ΔRR) | Écart-type des différences successives |
| `cv_rr` | SDNN / mean_RR | Coefficient de variation |

#### Domaine fréquentiel (4 features)

Calculées via **méthode de Welch** sur les RR interpolés à 4 Hz :

| Feature | Bande | Signification |
|---------|-------|---------------|
| `lf_power` | 0.04 – 0.15 Hz | Activité sympathique |
| `hf_power` | 0.15 – 0.40 Hz | Activité parasympathique |
| `lf_hf_ratio` | LF / HF | Balance autonomique |
| `total_power` | LF + HF | Puissance totale |

#### Domaine non-linéaire (3 features)

Analyse de Poincaré (RRn vs RRn+1) :

| Feature | Formule | Signification |
|---------|---------|---------------|
| `sd1` | std(RRn+1 - RRn) / √2 | Variabilité court-terme |
| `sd2` | std(RRn+1 + RRn) / √2 | Variabilité long-terme |
| `sd_ratio` | SD1 / SD2 | Ratio court/long terme |

---

### 4. Modèle IA

**Fichiers** : `app/ml/model.py`, `app/ml/inference.py`

#### Double mode de fonctionnement

Le modèle fonctionne en **deux modes** selon la disponibilité d'un modèle entraîné :

**Mode A — Modèle entraîné** (si `cognitive_model.joblib` + `scaler.joblib` existent) :
- Chargement sklearn via joblib
- Normalisation StandardScaler
- Prédiction : vecteur de 14 features → [stress, charge, fatigue]

**Mode B — Heuristique physiologique** (par défaut, sans modèle) :
- Basé sur la littérature HRV-cognition
- Fonctionne immédiatement, sans entraînement

```
STRESS = 0.4 × f(LF/HF) + 0.4 × f(RMSSD) + 0.2 × f(HR)
  → Activation sympathique élevée = stress élevé

CHARGE COGNITIVE = 0.35 × f(SDNN) + 0.35 × f(HR) + 0.3 × f(SD1)
  → HRV réduite + HR élevé = charge élevée

FATIGUE = 0.4 × f(RMSSD) + 0.35 × f(pNN50) + 0.25 × f(HR)
  → Retrait parasympathique = fatigue élevée
```

#### Lissage des scores
- **Méthode** : Moyenne mobile exponentielle (EMA)
- **Formule** : `score = 0.3 × brut + 0.7 × précédent`
- **Effet** : supprime les oscillations rapides, affichage stable

#### Prédiction de tendance fatigue
- Régression linéaire sur les ~120 dernières valeurs de fatigue
- Projection à **10 minutes** dans le futur
- **Confiance** = R² × facteur de couverture temporelle (0–1)

---

### 5. Stockage

**Fichiers** : `app/storage/database.py`, `app/storage/session_manager.py`

#### Base de données SQLite — 3 tables

**`sessions`** — Métadonnées des sessions d'enregistrement

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | TEXT PK | Identifiant unique (UUID 8 chars) |
| `start_time` | REAL | Timestamp début |
| `end_time` | REAL | Timestamp fin (null si active) |
| `activity_type` | TEXT | travail / etude / repos / autre |
| `status` | TEXT | active / completed |
| `notes` | TEXT | Notes libres |

**`data_points`** — Données brutes par fenêtre

| Colonne | Type | Description |
|---------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `session_id` | TEXT FK | Référence session |
| `timestamp` | REAL | Timestamp du point |
| `hr` | REAL | Fréquence cardiaque |
| `rmssd`, `sdnn`, `pnn50`, `mean_rr` | REAL | Features HRV |
| `lf_power`, `hf_power`, `lf_hf_ratio` | REAL | Features fréquentielles |
| `stress`, `cognitive_load`, `fatigue` | REAL | Scores cognitifs (0–100) |
| `window_quality` | REAL | Qualité du segment (0–1) |
| `fatigue_slope` | REAL | Pente de fatigue (pts/min) |
| `fatigue_predicted` | REAL | Fatigue prédite à +10 min |

**`session_summaries`** — Résumé calculé à la fin de chaque session

| Colonne | Type | Description |
|---------|------|-------------|
| `session_id` | TEXT PK | Référence session |
| `duration_sec` | REAL | Durée totale |
| `avg_hr`, `avg_rmssd` | REAL | Moyennes physiologiques |
| `avg_stress`, `avg_cognitive_load`, `avg_fatigue` | REAL | Moyennes des scores |
| `max_stress`, `max_cognitive_load`, `max_fatigue` | REAL | Valeurs maximales |
| `time_overload_pct` | REAL | % temps en surcharge (load > 70) |
| `time_recovery_pct` | REAL | % temps en récupération |
| `feedback` | TEXT | Feedback textuel auto-généré |

**Index** : `(session_id)`, `(session_id, timestamp)`, `(start_time)`

#### Gestion des sessions

- **Démarrage** : création UUID, enregistrement en DB, flag actif
- **Enregistrement** : chaque résultat d'inférence → insert data_point
- **Arrêt** : calcul du résumé, génération du feedback, sauvegarde
- **Export CSV** : toutes les colonnes de data_points
- **Export résumé** : session + summary en JSON

#### Feedback automatique

Généré à la fin de chaque session :

| Condition | Message |
|-----------|---------|
| Surcharge > 40% du temps | "Charge cognitive élevée pendant X% de la session." |
| Fatigue moyenne > 60 | "Fatigue mentale importante détectée." |
| Récupération > 30% | "Bons moments de récupération observés." |
| Stress moyen > 60 | "Niveau de stress élevé durant la session." |
| Aucun seuil dépassé | "Session dans les normes. Bon état cognitif général." |

---

### 6. Analyse & recommandations

**Fichier** : `app/domain/analysis_service.py`

#### Détection de périodes critiques

L'analyse identifie automatiquement 3 types de périodes dans chaque session :

| Type | Condition | Durée min |
|------|-----------|-----------|
| **Surcharge** | `cognitive_load > 70` en continu | 30s |
| **Récupération** | `stress < 30 ET fatigue < 30` en continu | 30s |
| **Fatigue prolongée** | `fatigue > 60` en continu | 30s |

#### Recommandations personnalisées

| Condition détectée | Recommandation |
|--------------------|----------------|
| Stress moyen > 60 | Respiration profonde (cohérence cardiaque 5-5-5) |
| Charge cognitive > 70 | Technique Pomodoro |
| Surcharge > 50% du temps | Pauses plus fréquentes |
| Fatigue moyenne > 60 | Pause longue ou changement d'activité |
| Récupération < 10% | Micro-pauses régulières |
| Tout va bien | "Bon équilibre cognitif" |

#### Historique & analytics

- **Digest journalier** : moyennes stress/charge/fatigue/HR par jour
- **Évolution hebdomadaire** : 7 jours glissants avec % de surcharge quotidien
- **Historique** : liste des 30 derniers jours avec résumé

---

## 🌐 API REST

**Base URL** : `http://localhost:5000/api`

### Santé

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Vérification du serveur |

### Capteur

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/device/status` | État du capteur (nom, batterie, qualité signal, connexion) |

### Monitoring

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/monitoring/status` | État du monitoring (connexion, session active, batterie, qualité signal) |

### Sessions

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/sessions` | Liste des sessions (`?limit=50&offset=0`) |
| `GET` | `/sessions/active` | Session active en cours |
| `GET` | `/sessions/:id` | Détails d'une session + résumé |
| `GET` | `/sessions/:id/data` | Tous les data points de la session |
| `GET` | `/sessions/:id/critical-periods` | Périodes de surcharge/récupération/fatigue |
| `GET` | `/sessions/:id/recommendations` | Recommandations personnalisées |

> **Note** : Le démarrage et l'arrêt des sessions se font automatiquement via WebSocket (`start_monitoring` / `stop_monitoring`). Il n'y a plus d'endpoints REST manuels pour start/stop.

### Export

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/sessions/:id/export/csv` | Télécharger le CSV complet |
| `GET` | `/sessions/:id/export/summary` | Résumé JSON exportable |

### Historique & analyse

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/history/days` | Résumé des 30 derniers jours (`?n=30`) |
| `GET` | `/history/:date` | Digest d'un jour (format `YYYY-MM-DD`) |
| `GET` | `/analysis/weekly` | Évolution hebdomadaire (`?end_date=2025-01-15`) |

### Paramètres

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/settings/window` | Taille et pas de fenêtre actuels |
| `PUT` | `/settings/window` | Modifier (`{"window_size_sec": 60, "window_step_sec": 10}`) |

---

## 🔌 WebSocket (temps réel)

**URL** : `http://localhost:5000` (Socket.IO)

### Événements Client → Serveur

| Événement | Description | Payload |
|-----------|-------------|---------|
| `start_monitoring` | Tout-en-un : scan → connect → stream → créer session | — |
| `stop_monitoring` | Tout-en-un : stop session → stop stream → disconnect | — |

Un seul événement `start_monitoring` remplace les anciens `scan_device`, `connect_device`, `start_streaming` et la création de session REST. Le monitoring démarre automatiquement dès la connexion du capteur.

### Événements Serveur → Client

| Événement | Description | Payload |
|-----------|-------------|---------|
| `monitoring_status` | Progression et état du monitoring | `{status, session?, summary?, reason?}` |
| `device_state` | État du capteur (à la connexion + changements) | `{connection_state, name, address, battery_level, signal_quality}` |
| `hr_update` | HR instantané (chaque battement) | `{hr, timestamp}` |
| `inference` | Résultat complet d'inférence (toutes les ~5s) | voir ci-dessous |
| `error` | Erreur | `{message}` |

### Payload `monitoring_status`

```json
{
  "status": "scanning | connecting | streaming | stopped",
  "session": { "id": "a1b2c3d4", "start_time": 1706123456.0, "activity_type": "autre", ... },
  "summary": { "avg_stress": 42.5, "avg_fatigue": 31.7, "feedback": "...", ... },
  "reason": "user_stopped | device_disconnected"
}
```

- `status` : progression du monitoring (`scanning` → `connecting` → `streaming` → `stopped`)
- `session` : présent quand le monitoring est actif (status = `streaming`)
- `summary` : présent quand le monitoring s'arrête (status = `stopped`), contient le résumé de session
- `reason` : présent à l'arrêt — `user_stopped` (arrêt manuel) ou `device_disconnected` (déconnexion inattendue)

### Payload `inference` (émis toutes les ~5 secondes)

```json
{
  "scores": {
    "stress": 42.5,
    "cognitive_load": 58.3,
    "fatigue": 31.7,
    "timestamp": 1706123456.789
  },
  "features": {
    "mean_hr": 72.4,
    "mean_rr": 829.0,
    "sdnn": 45.23,
    "rmssd": 38.56,
    "pnn50": 22.40,
    "sdsd": 35.12,
    "cv_rr": 0.0546,
    "lf_power": 125.80,
    "hf_power": 98.45,
    "lf_hf_ratio": 1.278,
    "total_power": 224.25,
    "sd1": 27.28,
    "sd2": 52.15,
    "sd_ratio": 0.523,
    "quality_ratio": 0.950,
    "sample_count": 36
  },
  "fatigue_trend": {
    "slope": 0.45,
    "predicted_fatigue_10min": 36.2,
    "confidence": 0.72
  },
  "timestamp": 1706123456.789,
  "window_quality": 0.950
}
```

---

## 🗄 Base de données

SQLite avec WAL mode et foreign keys activées.

```
┌────────────────┐      ┌──────────────────┐      ┌───────────────────────┐
│    sessions    │──1:N─│   data_points    │      │  session_summaries    │
│                │      │                  │      │                       │
│ id (PK)       │      │ id (PK)          │      │ session_id (PK, FK)   │
│ start_time    │      │ session_id (FK)   │      │ duration_sec          │
│ end_time      │      │ timestamp         │      │ avg_hr, avg_rmssd     │
│ activity_type │      │ hr, rmssd, sdnn   │      │ avg/max stress        │
│ status        │      │ pnn50, mean_rr    │      │ avg/max cognitive_load│
│ notes         │      │ lf/hf_power       │      │ avg/max fatigue       │
│               │      │ stress, load,     │      │ time_overload_pct     │
│               │      │ fatigue           │      │ time_recovery_pct     │
│               │──1:1─│ window_quality    │      │ feedback              │
│               │      │ fatigue_slope     │      │                       │
│               │      │ fatigue_predicted │      │                       │
└────────────────┘      └──────────────────┘      └───────────────────────┘
```

---

## ⚙ Configuration

Toutes les configurations sont centralisées dans `app/config/settings.py` via des dataclasses.

### BLE (capteur)

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `device_name` | "Polar Verity Sense" | Nom du capteur BLE |
| `scan_timeout` | 10s | Durée du scan |
| `reconnect_attempts` | 3 | Tentatives de reconnexion |
| `reconnect_delay` | 2s | Délai entre tentatives |

### Signal (traitement)

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `window_size_sec` | 30s | Taille de la fenêtre glissante |
| `window_step_sec` | 5s | Pas d'émission |
| `min_ppi_ms` | 300ms | PPI minimum physiologique |
| `max_ppi_ms` | 2000ms | PPI maximum physiologique |
| `max_ppi_diff_ratio` | 0.20 (20%) | Seuil de détection ectopique |
| `min_quality_ratio` | 0.80 (80%) | Qualité minimale acceptable |

### ML (intelligence artificielle)

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `score_smoothing_alpha` | 0.3 | Coefficient EMA (0=stable, 1=réactif) |
| `fatigue_horizon_min` | 10 min | Horizon de prédiction fatigue |

### Serveur

| Paramètre | Env var | Défaut | Description |
|-----------|---------|--------|-------------|
| `host` | `HOST` | 0.0.0.0 | Adresse d'écoute |
| `port` | `PORT` | 5000 | Port |
| `debug` | `DEBUG` | false | Mode debug |

---

## 🤖 Mode heuristique vs modèle entraîné

### Fonctionnement actuel (heuristique)

Au démarrage, le serveur cherche les fichiers :
```
app/ml/models/cognitive_model.joblib
app/ml/models/scaler.joblib
```

**S'ils n'existent pas** → mode heuristique activé automatiquement. Les scores sont calculés via des règles pondérées basées sur la littérature HRV :

- **Stress** ← LF/HF (40%) + RMSSD inversé (40%) + HR (20%)
- **Charge cognitive** ← SDNN inversé (35%) + HR (35%) + SD1 inversé (30%)
- **Fatigue** ← RMSSD inversé (40%) + pNN50 inversé (35%) + HR (25%)

Ce mode donne des résultats physiologiquement crédibles et permet d'utiliser l'application immédiatement.

### Passage au modèle entraîné

Pour passer en mode modèle entraîné :

1. Collecter des données avec l'application (sessions avec labels)
2. Exporter les CSV via `/api/sessions/:id/export/csv`
3. Entraîner un modèle sklearn (RandomForest, GradientBoosting, etc.)
4. Sauvegarder le modèle et le scaler :
   ```python
   import joblib
   joblib.dump(model, "app/ml/models/cognitive_model.joblib")
   joblib.dump(scaler, "app/ml/models/scaler.joblib")
   ```
5. Redémarrer le serveur → le modèle est chargé automatiquement

Le modèle doit accepter un vecteur de **14 features** en entrée et retourner **[stress, cognitive_load, fatigue]** (0–100).

---

## 📊 Flux de données Flutter ↔ Backend

```
FLUTTER                          BACKEND
───────                          ───────

  ┌─ emit("start_monitoring") ────→ Scan → Connect → Stream → Session
  │   ← on("monitoring_status")    {status: "scanning"}
  │   ← on("monitoring_status")    {status: "connecting"}
  │   ← on("monitoring_status")    {status: "streaming", session: {...}}
  │
  │   ┌─────── BOUCLE TEMPS RÉEL (automatique) ───┐
  │   │                                             │
  │   │  ← on("hr_update")    chaque battement     │
  │   │  ← on("inference")    toutes les ~5s       │
  │   │  ← on("device_state") si changement        │
  │   │                                             │
  │   │  Données enregistrées automatiquement       │
  │   │  en session (SQLite)                        │
  │   │                                             │
  │   └─────────────────────────────────────────────┘
  │
  ├─ emit("stop_monitoring") ─────→ Stop session → Disconnect
  │   ← on("monitoring_status")    {status: "stopped", summary: {...}}
  │
  │   OU déconnexion inattendue du capteur :
  │   ← on("monitoring_status")    {status: "stopped",
  │                                  reason: "device_disconnected",
  │                                  summary: {...}}
  │
  ├─ GET /monitoring/status ──────→ État courant du monitoring
  ├─ GET /sessions/:id/data ──────→ Courbes historiques
  ├─ GET /history/days ───────────→ Liste des jours
  ├─ GET /analysis/weekly ────────→ Évolution hebdo
  ├─ GET /sessions/:id/export/csv → Téléchargement CSV
  └───────────────────────────────
```

---

## 🔧 Notes techniques

- **Auto-monitoring** : un seul événement `start_monitoring` orchestre scan → connexion → streaming → création de session
- **Auto-stop** : en cas de déconnexion inattendue du capteur, la session est automatiquement arrêtée avec son résumé
- **Thread safety** : Socket.IO en mode `threading`, SQLite en mode WAL
- **Gestion d'erreurs** : chaque étape du pipeline est protégée, fallback gracieux
- **Reconnexion BLE** : 3 tentatives automatiques avec délai progressif
- **Qualité signal** : moyenne glissante sur 50 derniers flags de contact peau
- **Mémoire** : buffer glissant borné, historique fatigue limité à 120 points
