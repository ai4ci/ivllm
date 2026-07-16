import { loadCredentials, saveConfig } from '../config.ts';

export async function cmdConfig(
  host?: string,
  user?: string,
  path?: string,
  port?: string,
  token?: string,
): Promise<void> {
  const config = loadCredentials();
  if (host) config.loginHost = host!;
  if (user) config.username = user!;
  if (path) config.projectDir = path!;
  if (port) config.defaultLocalPort = parseInt(port!, 10);
  if (token) config.hfToken = token!;
  saveConfig(config);
  console.log('Configuration saved.');
}
