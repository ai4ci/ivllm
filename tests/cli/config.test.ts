import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { existsSync, rmSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { loadCredentials, saveConfig, assertConfigured } from "../src/config.ts";
import type { Credentials } from "../src/types.ts";

const REAL_CONFIG_PATH = join(homedir(), ".config", "ivllm", "config.json");

let savedConfig: string | null = null;

beforeEach(() => {
  savedConfig = existsSync(REAL_CONFIG_PATH) ? readFileSync(REAL_CONFIG_PATH, "utf-8") : null;
  if (existsSync(REAL_CONFIG_PATH)) rmSync(REAL_CONFIG_PATH);
});

afterEach(() => {
  if (existsSync(REAL_CONFIG_PATH)) rmSync(REAL_CONFIG_PATH);
  if (savedConfig !== null) writeFileSync(REAL_CONFIG_PATH, savedConfig, "utf-8");
});

describe("Config", () => {
  it("loadConfig returns defaults when no config file exists", () => {
    const config = loadCredentials();
    expect(config.defaultLocalPort).toBe(11434);
    expect(config.loginHost).toBe("");
    expect((config as unknown as Record<string, unknown>)["vllmVersion"]).toBeUndefined();
    expect((config as unknown as Record<string, unknown>)["venvPath"]).toBeUndefined();
  });

  it("saveConfig persists and loadConfig reads back", () => {
    const config = loadCredentials();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    saveConfig(config);

    const loaded = loadCredentials();
    expect(loaded.loginHost).toBe("login.isambard.ac.uk");
    expect(loaded.username).toBe("testuser");
    expect(loaded.defaultLocalPort).toBe(11434);
  });

  it("loadConfig merges saved values with defaults for missing fields", () => {
    const config = loadCredentials();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    saveConfig(config);

    // Simulate a config file missing a newer field — strip defaultLocalPort
    const raw = JSON.parse(readFileSync(REAL_CONFIG_PATH, "utf-8")) as Record<string, unknown>;
    delete raw["defaultLocalPort"];
    writeFileSync(REAL_CONFIG_PATH, JSON.stringify(raw));

    const merged = loadCredentials();
    expect(merged.defaultLocalPort).toBe(11434); // filled by default
    expect(merged.loginHost).toBe("login.isambard.ac.uk"); // preserved
  });

  it("assertConfigured throws when loginHost is missing", () => {
    const config = loadCredentials();
    expect(() => assertConfigured(config)).toThrow(/loginHost/);
  });

  it("assertConfigured throws when username is missing", () => {
    const config = loadCredentials();
    config.loginHost = "login.isambard.ac.uk";
    expect(() => assertConfigured(config)).toThrow(/username/);
  });

  it("assertConfigured does not throw when fully configured", () => {
    const config = loadCredentials();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    expect(() => assertConfigured(config)).not.toThrow();
  });

  it("hfToken is not present in defaults (optional field)", () => {
    const config = loadCredentials();
    expect((config as unknown as Record<string, unknown>)["hfToken"]).toBeUndefined();
  });

  it("hfToken round-trips through saveConfig/loadConfig", () => {
    const config = loadCredentials();
    (config as unknown as Record<string, unknown>)["hfToken"] = "hf_testtoken123";
    saveConfig(config as Credentials);
    const loaded = loadCredentials();
    expect((loaded as unknown as Record<string, unknown>)["hfToken"]).toBe("hf_testtoken123");
  });
});
