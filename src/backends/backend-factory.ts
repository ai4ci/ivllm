import { Backend } from './Backend';
import type { Credentials } from '../types';
import { IsambardBareMetalBackend } from './IsambardBareMetalBackend';

// ====== Backend registry =======

// Adding a backend:
// TODO: Will need to configure backend in credentials and create a wrapper
// structure for the config file that allows a list of credentials to be supplied
// will need to update config cli option to be an --backend option that
// takes all backend params at once and probably validates.
// Then need backend implementation (plus maybe RemoteOps) and add it here.

// Allowed configuration names
export type BackendType = 'isambard'; // | "isambard_container" | "local-llama-cpp"

// Registry Map to ensure compile-time exhaustive checks
const backendRegistry: Record<
    BackendType,
    new (creds: Credentials) => Backend
> = {
    isambard: IsambardBareMetalBackend,
};

// Build the Factory Function
/**
 * Create a backend instance from credentials.
 *
 * Looks up the implementation from the backend registry and instantiates it.
 * @param creds — SSH and HPC connection credentials
 * @returns A configured backend instance
 */
export function getBackend(creds: Credentials): Backend {
    // Runtime validation for the configuration string
    const backendType = 'isambard';
    // const backendType = creds.backend;
    // TODO: add in this when supporting mulitple backends
    if (!(backendType in backendRegistry)) {
        throw new Error(
            `Unsupported backend type: "${backendType}". Choose from: ${Object.keys(backendRegistry).join(', ')}`,
        );
    }

    const TargetBackend = backendRegistry[backendType as BackendType];
    return new TargetBackend(creds);
}
