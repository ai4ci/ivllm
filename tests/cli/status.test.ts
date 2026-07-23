import { describe, test, expect } from "bun:test";
import { parseStatusArgs, formatJobRow, formatJobTable } from "../src/commands/status.ts";
import type { JobDetails } from "../src/types.ts";

describe("parseStatusArgs", () => {
  test("no args returns undefined jobName", () => {
    expect(parseStatusArgs([])).toEqual({ jobName: undefined });
  });

  test("first positional arg is job name", () => {
    expect(parseStatusArgs(["myjob"])).toEqual({ jobName: "myjob" });
  });

  test("flag-only args return undefined jobName", () => {
    expect(parseStatusArgs(["--verbose"])).toEqual({ jobName: undefined });
  });
});

describe("formatJobRow", () => {
  test("running job includes status and model", () => {
    const job: JobDetails = {
      status: "running",
      job_name: "llm-job",
      slurm_job_id: "12345",
      compute_hostname: "gpu01",
      model: "Qwen/Qwen2.5-0.5B-Instruct",
      server_port: 8000,
    };
    const row = formatJobRow(job);
    expect(row).toContain("llm-job");
    expect(row).toContain("running");
    expect(row).toContain("12345");
    expect(row).toContain("Qwen/Qwen2.5-0.5B-Instruct");
  });

  test("failed job shows error", () => {
    const job: JobDetails = {
      status: "failed",
      job_name: "bad-job",
      slurm_job_id: "99999",
      error: "OOM on GPU",
    };
    const row = formatJobRow(job);
    expect(row).toContain("bad-job");
    expect(row).toContain("failed");
    expect(row).toContain("OOM on GPU");
  });

  test("pending job with minimal fields", () => {
    const job: JobDetails = {
      status: "pending",
      job_name: "new-job",
    };
    const row = formatJobRow(job);
    expect(row).toContain("new-job");
    expect(row).toContain("pending");
  });

  test("timeout job shows timeout status", () => {
    const job: JobDetails = {
      status: "timeout",
      job_name: "slow-job",
      slurm_job_id: "77777",
    };
    const row = formatJobRow(job);
    expect(row).toContain("slow-job");
    expect(row).toContain("timeout");
  });
});

describe("formatJobTable", () => {
  test("empty list shows no-jobs message", () => {
    const output = formatJobTable([]);
    expect(output.toLowerCase()).toContain("no");
  });

  test("single running job is included", () => {
    const jobs: JobDetails[] = [
      { status: "running", job_name: "solo", slurm_job_id: "111", model: "mymodel", server_port: 8000 },
    ];
    const output = formatJobTable(jobs);
    expect(output).toContain("solo");
    expect(output).toContain("running");
  });

  test("multiple jobs all appear in output", () => {
    const jobs: JobDetails[] = [
      { status: "running", job_name: "job-a", slurm_job_id: "1" },
      { status: "pending", job_name: "job-b", slurm_job_id: "2" },
      { status: "failed", job_name: "job-c", slurm_job_id: "3", error: "crash" },
    ];
    const output = formatJobTable(jobs);
    expect(output).toContain("job-a");
    expect(output).toContain("job-b");
    expect(output).toContain("job-c");
    expect(output).toContain("running");
    expect(output).toContain("pending");
    expect(output).toContain("failed");
  });

  test("table includes a header row", () => {
    const jobs: JobDetails[] = [
      { status: "running", job_name: "x" },
    ];
    const output = formatJobTable(jobs);
    // Header should contain column labels
    expect(output.toLowerCase()).toMatch(/job|name|status/);
  });
});
