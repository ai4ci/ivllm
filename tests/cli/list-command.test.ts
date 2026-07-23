import { describe, it, expect } from "bun:test";
import { cmdList } from "../src/commands/list.ts";

describe("cmdList", () => {
  it("--help prints usage", async () => {
    const logs: string[] = [];
    const originalLog = console.log;
    console.log = (msg: string) => logs.push(msg);
    await cmdList(["--help"]);
    console.log = originalLog;
    expect(logs.join("\n")).toContain("Usage: ivllm list");
  });
});
