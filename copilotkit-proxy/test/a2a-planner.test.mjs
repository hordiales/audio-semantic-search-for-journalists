import assert from "node:assert/strict";
import test from "node:test";

import { A2ASearchPlanner, buildSearchPlanPrompt, parseAgentSearchPlan } from "../dist/a2a-planner.js";
import { executeSearch, parseSearchRequest } from "../dist/direct-search.js";

const audioRequest = parseSearchRequest({
  query: "música",
  indexes: ["audio"],
  k: 10,
  rewrite: true,
});

const bothRequest = parseSearchRequest({
  query: "declaraciones sobre inflación",
  indexes: ["text", "audio"],
  k: 10,
  rewrite: true,
});

const textRequest = parseSearchRequest({
  query: "declaraciones sobre inflación",
  k: 10,
  rewrite: true,
});

const yamnetRequest = parseSearchRequest({
  query: "aplausos durante un discurso",
  indexes: ["yamnet"],
  k: 10,
  rewrite: true,
});

test("the A2A planning prompt requests only a JSON plan for the selected indexes", () => {
  const prompt = buildSearchPlanPrompt(audioRequest);

  assert.match(prompt, /DIRECT_SEARCH_PLANNER_V1/);
  assert.match(prompt, /"audio"/);
  assert.doesNotMatch(prompt, /"text_query"/);
  assert.match(prompt, /exactamente esa lista/);
  assert.match(prompt, /audio_query es obligatorio y el servicio la traduce al inglés/);
});

test("the A2A planner accepts a fenced JSON plan and preserves the user's indexes", () => {
  const fence = String.fromCharCode(96).repeat(3);
  const plan = parseAgentSearchPlan(
    fence + "json\n" +
      '{"original_query":"música","indexes":["audio"],"audio_query":"música instrumental","rationale":"Consulta acústica para CLAP."}' +
      "\n" + fence,
    audioRequest,
  );

  assert.deepEqual(plan, {
    original_query: "música",
    indexes: ["audio"],
    text_query: undefined,
    audio_query: "música instrumental",
    audio_query_en: undefined,
    yamnet_query: undefined,
    yamnet_query_en: undefined,
    rationale: "Consulta acústica para CLAP.",
  });
});

test("the A2A planner sends the planner protocol and validates its response", async () => {
  let sent;
  const planner = new A2ASearchPlanner("https://agent.example", {
    async sendMessage(message) {
      sent = message;
      return {
        result: {
          kind: "message",
          parts: [{
            kind: "text",
            text: '{"original_query":"música","indexes":["audio"],"audio_query":"música","rationale":"Consulta acústica."}',
          }],
        },
      };
    },
  });

  const plan = await planner.rewrite(audioRequest);

  assert.match(sent.message.parts[0].text, /DIRECT_SEARCH_PLANNER_V1/);
  assert.equal(plan.audio_query, "música");
});

test("the A2A planner rejects incomplete plans instead of changing the query silently", () => {
  assert.throws(
    () => parseAgentSearchPlan('{"indexes":["audio"]}', audioRequest),
    /audio_query/,
  );
});

test("the A2A planner cannot remove CLAP after the user enabled it", () => {
  assert.throws(
    () => parseAgentSearchPlan(
      '{"indexes":["text"],"text_query":"declaraciones sobre inflación económica","rationale":"La consulta busca contenido dicho."}',
      bothRequest,
    ),
    /debe coincidir/,
  );
});

test("search defaults to text and adds CLAP and YAMNet only through independent opt-ins", () => {
  assert.deepEqual(parseSearchRequest({ query: "inflación", k: 10, rewrite: false }).indexes, ["text"]);
  assert.deepEqual(
    parseSearchRequest({ query: "aplausos", include_clap: true, k: 10, rewrite: false }).indexes,
    ["text", "audio"],
  );
  assert.deepEqual(
    parseSearchRequest({ query: "aplausos", include_yamnet: true, k: 10, rewrite: false }).indexes,
    ["text", "yamnet"],
  );
  assert.deepEqual(
    parseSearchRequest({ query: "aplausos", include_clap: true, include_yamnet: true, k: 10, rewrite: false }).indexes,
    ["text", "audio", "yamnet"],
  );
});

test("CLAP search uses the service translation and records it in the effective plan", async () => {
  const request = parseSearchRequest({
    query: "aplausos",
    include_clap: true,
    k: 10,
    rewrite: false,
  });
  const calls = [];

  const result = await executeSearch(request, {
    async request(method, path, body) {
      calls.push({ method, path, body });
      return path === "/search/audio"
        ? { translated_query: "applause", results: [] }
        : { results: [] };
    },
  });

  assert.deepEqual(calls, [
    { method: "POST", path: "/search/semantic", body: { query: "aplausos", k: 10 } },
    { method: "POST", path: "/search/audio", body: { query: "aplausos", k: 10 } },
  ]);
  assert.equal(result.plan.audio_query_en, "applause");
  assert.equal(result.indexes.audio.translated_query, "applause");
});

test("the A2A planner uses the original query when a text-only plan omits a redundant text_query", () => {
  const plan = parseAgentSearchPlan(
    '{"indexes":["text"],"rationale":"La consulta busca contenido dicho."}',
    textRequest,
  );

  assert.equal(plan.text_query, textRequest.query);
});

test("YAMNet search uses its endpoint and records the service translation", async () => {
  const request = parseSearchRequest({
    query: "aplausos durante un discurso",
    include_yamnet: true,
    k: 5,
    rewrite: false,
  });
  const calls = [];

  const result = await executeSearch(request, {
    async request(method, path, body) {
      calls.push({ method, path, body });
      return path === "/search/yamnet"
        ? { translated_query: "applause during speech", results: [] }
        : { results: [] };
    },
  });

  assert.deepEqual(calls, [
    { method: "POST", path: "/search/semantic", body: { query: "aplausos durante un discurso", k: 5 } },
    { method: "POST", path: "/search/yamnet", body: { query: "aplausos durante un discurso", k: 5 } },
  ]);
  assert.equal(result.plan.yamnet_query_en, "applause during speech");
  assert.equal(result.indexes.yamnet.translated_query, "applause during speech");
});

test("the A2A planner requires a YAMNet query and cannot remove that source", () => {
  assert.throws(
    () => parseAgentSearchPlan('{"indexes":["yamnet"]}', yamnetRequest),
    /yamnet_query/,
  );
  assert.throws(
    () => parseAgentSearchPlan(
      '{"indexes":["text"],"text_query":"aplausos"}',
      yamnetRequest,
    ),
    /debe coincidir/,
  );
});
