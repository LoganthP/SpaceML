# Persistent deployment

The active target is **Vercel only**. Follow [DEPLOY-VERCEL.md](DEPLOY-VERCEL.md)
for Vercel Sandbox compute, Drive storage, authentication and verification.
No external backend URL is needed.

Existing Docker/Compose/dev commands remain available for local development.
`docker-compose.production.yml` is a local option, not the Vercel deployment.
The previous external-backend configuration generator has been removed.
