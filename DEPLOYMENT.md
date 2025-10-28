# Guide de Déploiement - Dagan AI Agent

Ce guide détaille la configuration de GitHub Actions et le déploiement sur VPS pour l'application Dagan AI Agent.

## 📋 Table des matières

1. [Configuration GitHub Actions](#1-configuration-github-actions)
2. [Configuration Supabase](#2-configuration-supabase)
3. [Préparation du VPS](#3-préparation-du-vps)
4. [Déploiement sur VPS](#4-déploiement-sur-vps)
5. [Configuration Nginx (Reverse Proxy)](#5-configuration-nginx-reverse-proxy)
6. [SSL/HTTPS avec Let's Encrypt](#6-sslhttps-avec-lets-encrypt)
7. [Maintenance et Monitoring](#7-maintenance-et-monitoring)

---

## 1. Configuration GitHub Actions

### 1.1 Configurer les Secrets GitHub

Les secrets sont nécessaires pour que la pipeline CI/CD fonctionne correctement.

**Étapes:**

1. Allez sur votre repository GitHub: `https://github.com/Isopope/DaganAIAgent`
2. Cliquez sur **Settings** → **Secrets and variables** → **Actions**
3. Cliquez sur **New repository secret**

**Secrets à ajouter:**

| Nom du Secret | Description | Exemple |
|--------------|-------------|---------|
| `VITE_API_URL` | URL publique de votre API backend | `https://api.votredomaine.com` |
| `DOCKER_USERNAME` | (Optionnel) Nom d'utilisateur Docker Hub | `votre-username` |
| `DOCKER_PASSWORD` | (Optionnel) Token Docker Hub | `dckr_pat_xxxxx` |

> **Note:** Les secrets `OPENAI_API_KEY`, `TAVILY_API_KEY`, etc. ne sont **pas** nécessaires dans GitHub Actions car ils seront configurés directement sur votre VPS.

### 1.2 Workflow CI/CD Actuel

La pipeline `.github/workflows/ci-cd.yml` effectue automatiquement:

**Sur chaque Push/Pull Request:**
- ✅ Test et lint du backend Python
- ✅ Build du frontend React
- ✅ Upload des artifacts de build

**Sur Push vers `main` uniquement:**
- ✅ Build de l'image Docker backend
- ✅ Build de l'image Docker frontend

### 1.3 (Optionnel) Push automatique vers Docker Hub

Si vous voulez que GitHub Actions pousse automatiquement les images vers Docker Hub:

**Étape 1: Créer un token Docker Hub**
```bash
# Connectez-vous sur https://hub.docker.com
# Allez dans Account Settings → Security → New Access Token
# Copiez le token généré
```

**Étape 2: Ajouter les secrets GitHub**
- `DOCKER_USERNAME`: votre nom d'utilisateur Docker Hub
- `DOCKER_PASSWORD`: le token créé à l'étape 1

**Étape 3: Modifier `.github/workflows/ci-cd.yml`**

Remplacez dans le job `build-backend-docker`:
```yaml
- name: Build Backend Docker image
  uses: docker/build-push-action@v5
  with:
    context: .
    file: ./Dockerfile
    push: true  # ← Changer false en true
    tags: ${{ secrets.DOCKER_USERNAME }}/dagan-backend:latest
    cache-from: type=gha,scope=backend
    cache-to: type=gha,mode=max,scope=backend
```

Et dans le job `build-frontend-docker`:
```yaml
- name: Build Frontend Docker image
  uses: docker/build-push-action@v5
  with:
    context: ./frontend
    file: ./frontend/Dockerfile
    push: true  # ← Changer false en true
    tags: ${{ secrets.DOCKER_USERNAME }}/dagan-frontend:latest
    cache-from: type=gha,scope=frontend
    cache-to: type=gha,mode=max,scope=frontend
```

Ajoutez aussi un step de login Docker avant les builds:
```yaml
- name: Login to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_PASSWORD }}
```

---

## 2. Configuration Supabase

L'application utilise **Supabase** comme base de données PostgreSQL hébergée avec l'extension pgvector.

### 2.1 Créer un Projet Supabase

1. Allez sur [https://supabase.com](https://supabase.com) et créez un compte
2. Cliquez sur **New Project**
3. Remplissez les informations:
   - **Name**: `dagan-ai-agent` (ou le nom de votre choix)
   - **Database Password**: Choisissez un mot de passe fort et **notez-le**
   - **Region**: Choisissez la région la plus proche de votre VPS
   - **Pricing Plan**: Free tier suffit pour commencer

4. Cliquez sur **Create new project** et attendez ~2 minutes

### 2.2 Activer l'Extension pgvector

Une fois votre projet créé:

1. Dans le menu latéral, cliquez sur **SQL Editor**
2. Créez une nouvelle requête et exécutez:

```sql
-- Activer l'extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Créer la table pour les embeddings
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id TEXT PRIMARY KEY,
    collection_id TEXT,
    embedding VECTOR(2000),
    document TEXT,
    cmetadata JSONB
);

-- Créer l'index pour les recherches vectorielles
CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_idx 
ON langchain_pg_embedding 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Créer la table pour les conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    tools_used TEXT[],
    vector_searches INTEGER DEFAULT 0,
    web_searches INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

3. Cliquez sur **Run** pour exécuter les commandes

### 2.3 Récupérer les Informations de Connexion

1. Dans le menu latéral, cliquez sur **Project Settings** (icône engrenage)
2. Allez dans **Database**
3. Dans la section **Connection String**, sélectionnez **URI** et copiez la chaîne de connexion
   - Elle ressemble à: `postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`
   - **Important**: Remplacez `[PASSWORD]` par le mot de passe de la base de données que vous avez noté à l'étape 2.1

4. Dans **Project Settings** → **API**, notez:
   - **Project URL** (SUPABASE_URL)
   - **anon public** key (SUPABASE_ANON_KEY)
   - **service_role secret** key (SUPABASE_SERVICE_KEY) - cliquez sur "Reveal" pour l'afficher

**Conservez précieusement ces informations pour la section 4.2**

---

## 3. Préparation du VPS

### 3.1 Prérequis VPS

**Spécifications minimales recommandées:**
- OS: Ubuntu 22.04 LTS ou Debian 11+
- RAM: 1 GB minimum (2 GB recommandé)
- Stockage: 10 GB minimum
- CPU: 1 core minimum (2 cores recommandé)

> **Note:** Les besoins sont réduits car la base de données est hébergée sur Supabase.

### 3.2 Connexion au VPS

```bash
# Remplacez par votre IP et utilisateur
ssh root@votre-ip-vps

# Ou si vous utilisez un utilisateur non-root
ssh votre-user@votre-ip-vps
```

### 3.3 Installation de Docker et Docker Compose

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Installation des dépendances
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Ajout de la clé GPG Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Ajout du repository Docker
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installation de Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Installation de Docker Compose v2
sudo apt install -y docker-compose-plugin

# Vérification des installations
docker --version
docker compose version

# Ajouter votre utilisateur au groupe docker (pour éviter sudo)
sudo usermod -aG docker $USER

# Déconnectez-vous et reconnectez-vous pour appliquer les changements
exit
ssh votre-user@votre-ip-vps
```

### 3.4 Installation de Git

```bash
sudo apt install -y git

# Vérification
git --version
```

---

## 4. Déploiement sur VPS

### 4.1 Cloner le Repository

```bash
# Créer un répertoire pour l'application
mkdir -p ~/apps
cd ~/apps

# Cloner le repository
git clone https://github.com/Isopope/DaganAIAgent.git
cd DaganAIAgent
```

### 4.2 Configuration des Variables d'Environnement

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer le fichier .env
nano .env
```

**Configurez les variables suivantes dans `.env`:**

```bash
# TAVILY Configuration
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=2000

# Supabase Configuration
SUPABASE_URL=https://xxxxxxxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxxxxxx
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxxxxxx

# PostgreSQL Connection (Supabase)
# Format: postgresql://postgres.[PROJECT-ID]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
POSTGRES_CONNECTION_STRING=postgresql://postgres.xxxxxxxxxx:VotreMotDePasse@aws-0-eu-central-1.pooler.supabase.com:6543/postgres

# Documents Collection
DOCUMENTS_COLLECTION=crawled_documents

# Frontend Configuration (URL publique de votre API)
# Laissez http://localhost:8000 pour le moment, vous le changerez après avoir configuré SSL
VITE_API_URL=http://localhost:8000
```

**Sauvegarder et quitter:**
- Appuyez sur `Ctrl+X`, puis `Y`, puis `Enter`

### 4.3 Lancer les Services avec Docker Compose

```bash
# Builder et lancer tous les services en arrière-plan
docker compose up -d --build

# Vérifier que les conteneurs sont démarrés
docker compose ps

# Suivre les logs
docker compose logs -f

# Pour sortir des logs: Ctrl+C
```

**Vous devriez voir 2 conteneurs:**
- `dagan-backend` (port 8000)
- `dagan-frontend` (port 80)

> **Note:** Plus de conteneur PostgreSQL car la base de données est hébergée sur Supabase.

### 4.4 Vérification du Déploiement

```bash
# Test de l'API backend
curl http://localhost:8000/health

# Réponse attendue:
# {"status":"everything is ok"}

# Test du frontend
curl http://localhost:80
# Vous devriez voir du HTML
```

> **Note:** L'initialisation de la base de données a déjà été effectuée dans la section 2.2 (Configuration Supabase).

---

## 5. Configuration Nginx (Reverse Proxy)

Pour exposer votre application sur Internet avec un nom de domaine.

### 5.1 Installation de Nginx sur le VPS

```bash
sudo apt install -y nginx

# Démarrer et activer Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Vérifier le statut
sudo systemctl status nginx
```

### 5.2 Configuration du Reverse Proxy

**Créer la configuration pour le backend (API):**

```bash
sudo nano /etc/nginx/sites-available/dagan-api
```

**Contenu du fichier:**

```nginx
server {
    listen 80;
    server_name api.votredomaine.com;  # ← Remplacer par votre domaine

    # Taille maximale des fichiers uploadés
    client_max_body_size 10M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeout pour les requêtes streaming
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

**Créer la configuration pour le frontend:**

```bash
sudo nano /etc/nginx/sites-available/dagan-frontend
```

**Contenu du fichier:**

```nginx
server {
    listen 80;
    server_name votredomaine.com www.votredomaine.com;  # ← Remplacer

    location / {
        proxy_pass http://localhost:80;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 5.3 Activer les Configurations

```bash
# Créer des liens symboliques
sudo ln -s /etc/nginx/sites-available/dagan-api /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/dagan-frontend /etc/nginx/sites-enabled/

# Tester la configuration
sudo nginx -t

# Recharger Nginx
sudo systemctl reload nginx
```

### 5.4 Configuration DNS

**Chez votre registrar de domaine (ex: OVH, Namecheap, etc.):**

Ajoutez ces enregistrements DNS:

| Type | Nom | Valeur | TTL |
|------|-----|--------|-----|
| A | @ | `IP_DE_VOTRE_VPS` | 3600 |
| A | api | `IP_DE_VOTRE_VPS` | 3600 |
| A | www | `IP_DE_VOTRE_VPS` | 3600 |

**Attendez la propagation DNS (peut prendre jusqu'à 24h, généralement ~1h).**

---

## 6. SSL/HTTPS avec Let's Encrypt

### 6.1 Installation de Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 6.2 Obtenir les Certificats SSL

```bash
# Pour le backend API
sudo certbot --nginx -d api.votredomaine.com

# Pour le frontend
sudo certbot --nginx -d votredomaine.com -d www.votredomaine.com
```

**Suivez les instructions:**
- Entrez votre email
- Acceptez les conditions
- Choisissez de rediriger HTTP vers HTTPS (option 2)

### 6.3 Renouvellement Automatique

Certbot configure automatiquement un cron job pour le renouvellement.

**Vérifier:**
```bash
sudo certbot renew --dry-run
```

### 6.4 Mise à Jour du fichier .env

```bash
cd ~/apps/DaganAIAgent
nano .env
```

**Modifier:**
```bash
VITE_API_URL=https://api.votredomaine.com  # ← HTTPS maintenant
```

**Reconstruire le frontend:**
```bash
docker compose up -d --build frontend
```

---

## 7. Maintenance et Monitoring

### 7.1 Commandes Docker Utiles

```bash
# Voir les logs en temps réel
docker compose logs -f

# Voir les logs d'un service spécifique
docker compose logs -f backend
docker compose logs -f frontend

# Redémarrer tous les services
docker compose restart

# Redémarrer un service spécifique
docker compose restart backend

# Arrêter tous les services
docker compose down

# Voir l'utilisation des ressources
docker stats
```

### 7.2 Mise à Jour de l'Application

```bash
cd ~/apps/DaganAIAgent

# Récupérer les dernières modifications
git pull origin main

# Reconstruire et redémarrer les services
docker compose up -d --build

# Vérifier que tout fonctionne
docker compose ps
docker compose logs -f
```

### 7.3 Backup de la Base de Données Supabase

**Supabase propose des backups automatiques**, mais vous pouvez aussi faire des backups manuels.

**Option 1: Backup via Supabase Dashboard**
1. Allez dans votre projet Supabase
2. **Database** → **Backups**
3. Cliquez sur **Download** pour un backup existant ou **Start a backup** pour en créer un nouveau

**Option 2: Backup manuel via pg_dump**
```bash
# Installer postgresql-client si nécessaire
sudo apt install -y postgresql-client

# Créer un backup (remplacez par votre connection string)
pg_dump "postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:6543/postgres" > backup_$(date +%Y%m%d_%H%M%S).sql

# Ou avec compression
pg_dump "postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:6543/postgres" | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

**Restaurer un backup:**
```bash
# Restaurer (⚠️ Attention: écrase les données existantes)
psql "postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:6543/postgres" < backup_20250128_120000.sql

# Ou si compressé
gunzip -c backup_20250128_120000.sql.gz | psql "postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:6543/postgres"
```

### 7.4 Monitoring des Logs

```bash
# Installer logrotate si pas déjà installé
sudo apt install -y logrotate

# Les logs Docker sont automatiquement gérés
# Pour voir la taille des logs:
docker inspect --format='{{.LogPath}}' dagan-backend
docker inspect --format='{{.LogPath}}' dagan-frontend
```

### 7.5 Surveillance des Ressources

```bash
# Installer htop pour monitoring en temps réel
sudo apt install -y htop

# Lancer htop
htop

# Voir l'espace disque
df -h

# Voir l'utilisation mémoire
free -h

# Voir les processus Docker
docker stats --no-stream
```

---

## 8. Dépannage

### 8.1 Le backend ne démarre pas

**Vérifier les logs:**
```bash
docker compose logs backend
```

**Problèmes courants:**
- Variables d'environnement manquantes → Vérifier `.env`
- `POSTGRES_CONNECTION_STRING` invalide → Vérifier la chaîne de connexion Supabase
- Supabase inaccessible → Vérifier que votre projet Supabase est actif
- Port 8000 déjà utilisé → `sudo lsof -i :8000` puis tuer le processus

### 8.2 Le frontend ne se connecte pas à l'API

**Vérifier:**
1. `VITE_API_URL` dans `.env` est correct
2. Reconstruire le frontend après modification: `docker compose up -d --build frontend`
3. Vérifier les CORS dans le backend

### 8.3 Erreurs de connexion à Supabase

**Vérifier:**
1. La chaîne de connexion `POSTGRES_CONNECTION_STRING` est correcte
2. Le mot de passe ne contient pas de caractères spéciaux non échappés
3. Votre IP n'est pas bloquée par Supabase (vérifier les paramètres réseau du projet)
4. L'extension pgvector est bien activée (voir section 2.2)

**Test de connexion:**
```bash
# Installer postgresql-client
sudo apt install -y postgresql-client

# Tester la connexion (remplacez par votre connection string)
psql "postgresql://postgres.xxxxx:password@aws-0-region.pooler.supabase.com:6543/postgres" -c "SELECT version();"
```

### 8.4 Erreurs de mémoire

**Augmenter la limite mémoire Docker:**
```bash
# Éditer docker-compose.yml et ajouter:
services:
  backend:
    mem_limit: 2g
  frontend:
    mem_limit: 512m
```

---

## 9. Checklist de Déploiement

- [ ] Projet Supabase créé avec extension pgvector activée
- [ ] Tables de base de données créées dans Supabase
- [ ] Informations de connexion Supabase récupérées (URL, keys, connection string)
- [ ] VPS configuré avec Docker et Docker Compose
- [ ] Repository cloné sur le VPS
- [ ] Fichier `.env` créé et configuré avec toutes les clés API et Supabase
- [ ] Services Docker lancés (`docker compose up -d --build`)
- [ ] Nginx installé et configuré en reverse proxy
- [ ] DNS configuré (enregistrements A pour domaine et api.domaine)
- [ ] Certificats SSL obtenus avec Certbot
- [ ] `VITE_API_URL` mis à jour avec HTTPS
- [ ] Frontend reconstruit avec la nouvelle variable
- [ ] Tests de l'application (frontend + backend)
- [ ] Backup Supabase vérifié

---

## 10. Ressources Utiles

- **Documentation Docker**: https://docs.docker.com
- **Documentation Nginx**: https://nginx.org/en/docs
- **Let's Encrypt**: https://letsencrypt.org
- **GitHub Actions**: https://docs.github.com/en/actions
- **Supabase Documentation**: https://supabase.com/docs
- **Supabase pgvector**: https://supabase.com/docs/guides/ai/vector-columns

---

**🎉 Félicitations ! Votre application Dagan AI Agent est maintenant déployée en production.**

Pour toute question ou problème, consultez les logs avec `docker compose logs -f`.
