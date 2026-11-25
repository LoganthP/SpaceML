// Vercel assigns each service a port. Read it without platform-specific shell
// expansion so `vercel dev` also works on Windows.
const path = require('node:path');
const { pathToFileURL } = require('node:url');
async function main() {
  const root = path.resolve(__dirname, '../frontend');
  const { createServer } = await import(pathToFileURL(path.join(root, 'node_modules/vite/dist/node/index.js')).href);
  const server = await createServer({
    root, configFile: path.join(root, 'vite.config.ts'),
    server: { host: '127.0.0.1', port: Number(process.env.PORT || 5173), strictPort: true },
  });
  await server.listen();
  server.printUrls();
}
main().catch(error => { console.error(error); process.exitCode = 1; });
