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

test("the A2A planning prompt requests only a JSON plan for the selected indexes", () => {
  const prompt = buildSearchPlanPrompt(audioRequest);

  assert.match(prompt, /DIRECT_SEARCH_PLANNER_V1/);
  assert.match(prompt, /"audio"/);
  assert.doesNotMatch(prompt, /"text_query"/);
  assert.match(prompt, /Elegí ambos sólo/);
  assert.match(prompt, /text_query es obligatorio/);
});

test("the A2A planner accepts a fenced JSON plan and preserves the user's indexes", () => {
  const fence = String.fromCharCode(96).repeat(3);
  const plan = parseAgentSearchPlan(
    fence + "json\n" +
      '{"original_query":"música","indexes":["audio"],"audio_query":"música instrumental","audio_query_en":"instrumental music","rationale":"La descripción acústica en inglés mejora CLAP."}' +
      "\n" + fence,
    audioRequest,
  );

  assert.deepEqual(plan, {
    original_query: "música",
    indexes: ["audio"],
    text_query: undefined,
    audio_query: "música instrumental",
    audio_query_en: "instrumental music",
    rationale: "La descripción acústica en inglés mejora CLAP.",
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
            text: '{"original_query":"música","indexes":["audio"],"audio_query":"música","audio_query_en":"music","rationale":"Consulta acústica."}',
          }],
        },
      };
    },
  });

  const plan = await planner.rewrite(audioRequest);

  assert.match(sent.message.parts[0].text, /DIRECT_SEARCH_PLANNER_V1/);
  assert.equal(plan.audio_query_en, "music");
});

test("the A2A planner rejects incomplete plans instead of changing the query silently", () => {
  assert.throws(
    () => parseAgentSearchPlan('{"audio_query":"música"}', audioRequest),
    /audio_query_en/,
  );
});

test("the A2A planner may select one enabled index and retrieval follows that selection", async () => {
  const plan = parseAgentSearchPlan(
    '{"indexes":["text"],"text_query":"declaraciones sobre inflación económica","rationale":"La consulta busca contenido dicho."}',
    bothRequest,
  );
  const calls = [];

  const result = await executeSearch({ ...bothRequest, plan }, {
    async request(method, path, body) {
      calls.push({ method, path, body });
      return { results: [] };
    },
  });

  assert.deepEqual(plan.indexes, ["text"]);
  assert.deepEqual(calls, [{
    method: "POST",
    path: "/search/semantic",
    body: { query: "declaraciones sobre inflación económica", k: 10 },
  }]);
  assert.deepEqual(Object.keys(result.indexes), ["text"]);
});

test("the A2A planner uses the original query when a text-only plan omits a redundant text_query", () => {
  const plan = parseAgentSearchPlan(
    '{"indexes":["text"],"rationale":"La consulta busca contenido dicho."}',
    bothRequest,
  );

  assert.equal(plan.text_query, bothRequest.query);
});
