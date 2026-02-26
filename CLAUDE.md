# CLAUDE.md — CI/CD & Deployment Instructies

## Overzicht

Dit project gebruikt een geautomatiseerde CI/CD pipeline:

```
Claude Code → GitHub (push/merge) → GitHub Actions (build & push image) → Tailscale → Portainer API → Synology Docker update
```

## Architectuur

- **Code:** GitHub repository
- **Container Registry:** GitHub Container Registry (ghcr.io)
- **CI/CD:** GitHub Actions (build on merge to main)
- **Netwerk:** Tailscale (GitHub Actions runner joins tailnet om Synology te bereiken)
- **Orchestratie:** Portainer stacks op Synology NAS (Community Edition)
- **Runtime:** Docker op Synology

## Hoe secrets werken

Secrets (API keys, tokens) worden **nooit** in de git repo opgeslagen. In plaats daarvan:

1. Secrets staan als **GitHub Secrets** op de repository
2. De GitHub Actions workflow leest de docker-compose.yml uit de repo
3. De workflow stuurt de compose-inhoud + secrets via de **Portainer API** naar de Synology
4. Portainer injecteert de secrets als environment variables in de container

Hierdoor hoef je in Portainer zelf niets in te stellen — alles wordt beheerd via GitHub.

### Secrets in docker-compose.yml

Gebruik `${VARIABLE_NAME}` syntax voor secrets. Deze worden door Portainer vervangen:

```yaml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}    # Komt uit GitHub Secrets → Portainer env
  - PHOTOS_PATHS=/volume1/photos         # Vast pad, geen secret nodig
```

### Secrets toevoegen aan de deploy workflow

In `.github/workflows/deploy.yml`, voeg de secret toe aan de `PAYLOAD` in de "Update Portainer stack" stap:

```yaml
PAYLOAD=$(jq -n \
  --arg content "$COMPOSE_CONTENT" \
  --arg openai_key "${{ secrets.OPENAI_API_KEY }}" \
  --arg new_secret "${{ secrets.NEW_SECRET }}" \
  '{
    stackFileContent: $content,
    env: [
      { name: "OPENAI_API_KEY", value: $openai_key },
      { name: "NEW_SECRET", value: $new_secret }
    ],
    prune: true,
    pullImage: true
  }')
```

## Vereisten per repository

### 1. Dockerfile
Elke repository MOET een Dockerfile in de root hebben.

### 2. GitHub Actions workflow
Kopieer `.github/workflows/deploy.yml` naar elke nieuwe repository. De workflow:
1. Bouwt het Docker image
2. Pusht naar ghcr.io
3. Verbindt met Tailscale
4. Stuurt docker-compose.yml + secrets naar Portainer API om stack te updaten

### 3. .dockerignore
Elke repository moet een `.dockerignore` hebben met minimaal:
```
.git
.github
.env
stack.env
*.md
__pycache__
*.pyc
.pytest_cache
.venv
tests
docker-compose*.yml
```

### 4. docker-compose.yml (Portainer stack)
Per project een docker-compose.yml met `image: ghcr.io/gjagils/<REPO_NAME>:latest`.
Gebruik `${VAR}` syntax voor secrets die via de Portainer API worden geïnjecteerd.

## GitHub Secrets (per repository)

| Secret | Waarde | Herbruikbaar? |
|--------|--------|---------------|
| `TAILSCALE_AUTHKEY` | Tailscale auth key (reusable + ephemeral) | Ja, zelfde voor alle repos |
| `PORTAINER_API_TOKEN` | Portainer API access token | Ja, zelfde voor alle repos |
| `PORTAINER_URL` | `http://100.65.249.84:9000` (Tailscale IP) | Ja, zelfde voor alle repos |
| `PORTAINER_ENDPOINT_ID` | Endpoint ID uit Portainer | Ja, zelfde voor alle repos |
| `PORTAINER_STACK_ID` | Stack ID uit Portainer URL | **Nee, uniek per project** |
| `OPENAI_API_KEY` | OpenAI API key (indien nodig) | Per project, indien nodig |

### Secrets instellen via CLI
```bash
# Herbruikbare secrets (zelfde voor elk project)
gh secret set TAILSCALE_AUTHKEY --body "<KEY>" --repo gjagils/<REPO>
gh secret set PORTAINER_API_TOKEN --body "<TOKEN>" --repo gjagils/<REPO>
gh secret set PORTAINER_URL --body "http://100.65.249.84:9000" --repo gjagils/<REPO>
gh secret set PORTAINER_ENDPOINT_ID --body "3" --repo gjagils/<REPO>

# Uniek per project
gh secret set PORTAINER_STACK_ID --body "<ID>" --repo gjagils/<REPO>

# Project-specifieke secrets (indien nodig)
gh secret set OPENAI_API_KEY --body "<KEY>" --repo gjagils/<REPO>
```

## Setup nieuw project (checklist)

1. [ ] Dockerfile in de root
2. [ ] `.github/workflows/deploy.yml` kopiëren en image-naam aanpassen
3. [ ] `.dockerignore` aanmaken
4. [ ] `docker-compose.yml` maken met `image: ghcr.io/gjagils/<REPO>:latest`
5. [ ] Eerste push naar main (triggert image build op ghcr.io)
6. [ ] GitHub Packages visibility instellen
7. [ ] Portainer stack **handmatig** aanmaken (eerste keer via Web editor met de docker-compose.yml)
8. [ ] Stack ID noteren uit Portainer URL
9. [ ] GitHub secrets instellen (5 herbruikbaar + stack ID + project-specifieke secrets)
10. [ ] Test: push een wijziging naar main en controleer of de stack automatisch update

## Commit conventie

Gebruik Conventional Commits:
- `feat:` — nieuwe feature
- `fix:` — bugfix
- `docs:` — documentatie
- `chore:` — onderhoud, dependencies
- `refactor:` — code refactoring

## Workflow voor Claude Code

```bash
# 1. Maak een branch
git checkout -b feature/beschrijving

# 2. Maak wijzigingen en commit
git add .
git commit -m "feat: beschrijving van de wijziging"

# 3. Push en maak PR
git push -u origin feature/beschrijving
gh pr create --title "feat: beschrijving" --body "Beschrijving van de wijziging"

# 4. Merge de PR (triggert automatisch build + deploy)
gh pr merge --squash
```

## Troubleshooting

### Image wordt niet gepulld op Synology
```bash
docker pull ghcr.io/gjagils/<REPO>:latest
docker login ghcr.io
```

### GitHub Actions faalt
- Check of GITHUB_TOKEN permissions `packages: write` heeft (staat in de workflow)
- Check of de Dockerfile geldig is: `docker build -t test .`

### Tailscale connect faalt
- Check of de auth key nog geldig is op https://login.tailscale.com/admin/settings/keys
- Maak eventueel een nieuwe reusable + ephemeral key aan

### Portainer API update faalt
- Check of het stack ID klopt (kijk in de Portainer URL)
- Test handmatig: `curl -s -H "X-API-Key: <TOKEN>" http://100.65.249.84:9000/api/stacks/<ID>`
- Check of endpointId correct is

### Container start niet
```bash
docker logs <CONTAINER_NAME>
docker inspect ghcr.io/gjagils/<REPO>:latest
```
