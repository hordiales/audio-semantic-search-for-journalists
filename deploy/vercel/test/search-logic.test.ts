import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { effectivePlan, parseSearchRequest } from "../src/direct-search.js";
import { normalizedAudioClassTokens } from "../src/db.js";
import { toPgVectorLiteral } from "../src/embeddings.js";
import { resolveAllowedOrigin } from "../src/cors.js";

describe("parseSearchRequest", () => {
  test("maps include_* flags to indexes (frontend-direct-search contract)", () => {
    const parsed = parseSearchRequest({
      query: "aplausos durante el discurso",
      include_text: true,
      include_clap: true,
      include_yamnet: false,
      k: 10,
      rewrite: false,
    });
    assert.deepEqual(parsed.indexes, ["text", "audio"]);
    assert.equal(parsed.query, "aplausos durante el discurso");
    assert.equal(parsed.k, 10);
    assert.equal(parsed.rewrite, false);
  });

  test("defaults to text index only when flags are omitted", () => {
    const parsed = parseSearchRequest({ query: "hola" });
    assert.deepEqual(parsed.indexes, ["text"]);
    assert.equal(parsed.k, 10);
    assert.equal(parsed.rewrite, true);
  });

  test("accepts an explicit indexes array and dataset_id", () => {
    const parsed = parseSearchRequest({
      query: "música",
      indexes: ["yamnet", "audio"],
      dataset_id: "archive-2024",
      k: 5,
    });
    assert.deepEqual(parsed.indexes, ["yamnet", "audio"]);
    assert.equal(parsed.dataset_id, "archive-2024");
  });

  test("rejects an empty query", () => {
    assert.throws(() => parseSearchRequest({ query: "   " }));
    assert.throws(() => parseSearchRequest({}));
    assert.throws(() => parseSearchRequest(null));
  });

  test("rejects when no index is selected", () => {
    assert.throws(() =>
      parseSearchRequest({
        query: "hola",
        include_text: false,
        include_clap: false,
        include_yamnet: false,
      }),
    );
  });

  test("rejects an out-of-range k", () => {
    assert.throws(() => parseSearchRequest({ query: "hola", k: 0 }));
    assert.throws(() => parseSearchRequest({ query: "hola", k: 51 }));
    assert.throws(() => parseSearchRequest({ query: "hola", k: 2.5 }));
  });
});

describe("effectivePlan", () => {
  test("returns the authored plan stamped with the original query", () => {
    const plan = {
      original_query: "vieja",
      indexes: ["text" as const],
      text_query: "editada",
      rationale: "manual",
    };
    const result = effectivePlan({
      query: "nueva consulta",
      indexes: ["text"],
      k: 10,
      rewrite: true,
      plan,
    });
    assert.equal(result.original_query, "nueva consulta");
    assert.equal(result.text_query, "editada");
  });

  test("builds a literal plan per enabled index when none was sent", () => {
    const result = effectivePlan({
      query: "aplausos",
      indexes: ["text", "audio", "yamnet"],
      k: 10,
      rewrite: true,
    });
    assert.equal(result.text_query, "aplausos");
    assert.equal(result.audio_query, "aplausos");
    assert.equal(result.yamnet_query, "aplausos");
    assert.deepEqual(result.indexes, ["text", "audio", "yamnet"]);
  });
});

describe("normalizedAudioClassTokens", () => {
  test("lowercases, strips accents and removes English stopwords", () => {
    assert.deepEqual(normalizedAudioClassTokens("Música y Aplausos en el audio"), [
      "aplausos",
      "el",
      "en",
      "musica",
      "y",
    ]);
  });

  test("produces stable sorted unique tokens for AudioSet matching", () => {
    const tokens = normalizedAudioClassTokens("Applause, applause during the speech!");
    assert.ok(tokens.includes("applause"));
    assert.ok(tokens.includes("speech"));
    assert.equal(new Set(tokens).size, tokens.length);
    assert.deepEqual([...tokens].sort(), tokens);
  });

  test("returns an empty list for stopword-only input", () => {
    assert.deepEqual(normalizedAudioClassTokens("the sound of audio"), []);
  });
});

describe("toPgVectorLiteral", () => {
  test("formats a pgvector literal with fixed precision", () => {
    assert.equal(toPgVectorLiteral([0.1, 0.5, -0.25]), "[0.10000000,0.50000000,-0.25000000]");
    assert.equal(toPgVectorLiteral([]), "[]");
  });
});

describe("resolveAllowedOrigin", () => {
  test("returns undefined when ALLOWED_ORIGINS is not configured", () => {
    assert.equal(resolveAllowedOrigin("https://example.com", undefined), undefined);
    assert.equal(resolveAllowedOrigin("https://example.com", ""), undefined);
    assert.equal(resolveAllowedOrigin("https://example.com", "  "), undefined);
  });

  test("reflects the request origin when it is in the allow list", () => {
    const allowed = "https://a.example, https://b.example";
    assert.equal(resolveAllowedOrigin("https://a.example", allowed), "https://a.example");
    assert.equal(resolveAllowedOrigin("https://b.example", allowed), "https://b.example");
  });

  test("supports the '*' wildcard", () => {
    assert.equal(resolveAllowedOrigin("https://any.example", "*"), "*");
  });

  test("rejects origins outside the allow list or missing Origin header", () => {
    const allowed = "https://a.example";
    assert.equal(resolveAllowedOrigin("https://evil.example", allowed), undefined);
    assert.equal(resolveAllowedOrigin(undefined, allowed), undefined);
  });
});
