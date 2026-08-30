import { type CSSProperties, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { search } from "./api";
import type { IndexResults, IndexSelection, SearchIndex, SearchPlan, SearchResponse, SearchResult } from "./types";

const INDEXES: SearchIndex[] = ["text", "audio"];
const labels: Record<SearchIndex, string> = { text: "Índice de texto", audio: "Índice de audio (CLAP)" };
const labelShort: Record<SearchIndex, string> = { text: "Texto", audio: "Audio" };

const initialUrl = new URLSearchParams(window.location.search);
const initialIndex = (initialUrl.get("idx") ?? "both") as IndexSelection;
const selectedFrom = (selection: IndexSelection): SearchIndex[] => selection === "both" ? INDEXES : [selection];
const asTime = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
};
const resultKey = (index: SearchIndex, result: SearchResult) => `${index}-${result.segment_id}-${result.rank}`;

function weakThreshold(index: SearchIndex): number | undefined {
  const source = index === "text" ? import.meta.env.VITE_WEAK_TEXT_SIMILARITY : import.meta.env.VITE_WEAK_AUDIO_SIMILARITY;
  const value = Number(source);
  return Number.isFinite(value) && source !== undefined ? value : undefined;
}

interface PlayerCommand { play: () => void; focus: () => void }

export default function App() {
  const [query, setQuery] = useState(initialUrl.get("q") ?? "");
  const [selection, setSelection] = useState<IndexSelection>(INDEXES.includes(initialIndex as SearchIndex) ? initialIndex : "both");
  const [k, setK] = useState(Number(initialUrl.get("k")) || 10);
  const [rewrite, setRewrite] = useState(true);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingPlan, setEditingPlan] = useState(false);
  const [draftPlan, setDraftPlan] = useState<SearchPlan | null>(null);
  const [fileFilter, setFileFilter] = useState("");
  const [fromTime, setFromTime] = useState("");
  const [toTime, setToTime] = useState("");
  const activeAudio = useRef<HTMLAudioElement | null>(null);
  const playerCommands = useRef(new Map<string, PlayerCommand>());
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
  const [queue, setQueue] = useState<string[] | null>(null);
  const controller = useRef<AbortController | null>(null);

  const selectedIndexes = selectedFrom(selection);
  const allResults = useMemo(() => selectedIndexes.flatMap(index =>
    (response?.indexes[index]?.results ?? []).map(result => ({ index, result })),
  ), [response, selectedIndexes]);

  useEffect(() => () => controller.current?.abort(), []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (query.trim()) url.searchParams.set("q", query.trim()); else url.searchParams.delete("q");
    url.searchParams.set("idx", selection);
    url.searchParams.set("k", String(k));
    window.history.replaceState(null, "", url);
  }, [query, selection, k]);

  const performSearch = useCallback(async (plan?: SearchPlan) => {
    const cleanedQuery = query.trim();
    if (!cleanedQuery) {
      setError("Escribí una consulta para buscar en el archivo.");
      return;
    }
    controller.current?.abort();
    const nextController = new AbortController();
    controller.current = nextController;
    setLoading(true);
    setError(null);
    setQueue(null);
    try {
      const result = await search({ query: cleanedQuery, indexes: selectedIndexes, k, rewrite, plan }, nextController.signal);
      setResponse(result);
      setDraftPlan(result.plan);
      setEditingPlan(false);
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") setError((reason as Error).message);
    } finally {
      if (controller.current === nextController) setLoading(false);
    }
  }, [k, query, rewrite, selectedIndexes]);

  const onSubmit = (event: FormEvent) => { event.preventDefault(); void performSearch(); };

  useEffect(() => {
    if (!response) return;
    const expirations = Object.values(response.indexes)
      .flatMap(bucket => bucket?.results ?? [])
      .map(result => result.clip_expires_at ? new Date(result.clip_expires_at).getTime() : Number.NaN)
      .filter(Number.isFinite);
    if (!expirations.length) return;
    // Renew slightly before the signed URL expires, while the current plan still
    // represents the exact retrieval the journalist is reviewing.
    const delay = Math.max(1_000, Math.min(...expirations) - Date.now() - 30_000);
    const timer = window.setTimeout(() => void performSearch(response.plan), delay);
    return () => window.clearTimeout(timer);
  }, [performSearch, response]);

  const requestPlay = useCallback((audio: HTMLAudioElement) => {
    if (activeAudio.current && activeAudio.current !== audio) activeAudio.current.pause();
    activeAudio.current = audio;
  }, []);
  const registerPlayer = useCallback((key: string, command: PlayerCommand | null) => {
    if (command) playerCommands.current.set(key, command); else playerCommands.current.delete(key);
  }, []);
  const onPlayerFinished = useCallback((key: string) => {
    setQueue(current => {
      if (!current) return current;
      const position = current.indexOf(key);
      const next = current[position + 1];
      if (!next) return null;
      window.setTimeout(() => playerCommands.current.get(next)?.play(), 0);
      return current;
    });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select, button")) return;
      if (!allResults.length) return;
      const current = Math.max(0, allResults.findIndex(item => resultKey(item.index, item.result) === focusedKey));
      if (event.key === " " || event.code === "Space") {
        event.preventDefault();
        const key = resultKey(allResults[current].index, allResults[current].result);
        playerCommands.current.get(key)?.play();
      }
      if (event.key.toLowerCase() === "j" || event.key.toLowerCase() === "k") {
        event.preventDefault();
        const offset = event.key.toLowerCase() === "j" ? 1 : -1;
        const targetResult = allResults[Math.min(Math.max(current + offset, 0), allResults.length - 1)];
        const key = resultKey(targetResult.index, targetResult.result);
        setFocusedKey(key);
        playerCommands.current.get(key)?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [allResults, focusedKey]);

  const savePlan = () => {
    if (!draftPlan) return;
    const plan = { ...draftPlan, original_query: query, indexes: selectedIndexes };
    setDraftPlan(plan);
    void performSearch(plan);
  };
  const startContinuous = () => {
    const keys = allResults.map(item => resultKey(item.index, item.result));
    if (keys.length) {
      setQueue(keys);
      playerCommands.current.get(keys[0])?.play();
    }
  };

  return (
    <main className="shell">
      <header className="masthead">
        <p className="eyebrow">Archivo sonoro · búsqueda directa</p>
        <h1>Encontrá y escuchá la evidencia.</h1>
        <p className="lede">Los resultados vienen completos del índice. La IA sólo puede proponer cómo buscar.</p>
      </header>

      <section className="search-panel" aria-label="Búsqueda">
        <form onSubmit={onSubmit}>
          <div className="search-line">
            <label className="sr-only" htmlFor="query">Consulta</label>
            <input id="query" value={query} onChange={event => setQuery(event.target.value)} placeholder="¿Hay música mientras habla el entrevistado?" autoFocus />
            <button className="primary" type="submit" disabled={loading}>{loading ? "Buscando…" : "Buscar"}</button>
          </div>
          <div className="controls-row">
            <fieldset className="index-picker">
              <legend>{rewrite ? "Índices habilitados para IA" : "Índice"}</legend>
              {(["both", "text", "audio"] as IndexSelection[]).map(option => (
                <label key={option}>
                  <input type="radio" checked={selection === option} onChange={() => setSelection(option)} />
                  {option === "both" ? "Ambos" : labelShort[option]}
                </label>
              ))}
            </fieldset>
            <label className="toggle"><input type="checkbox" checked={rewrite} onChange={event => setRewrite(event.target.checked)} /> Reformular con IA</label>
            <label className="k-picker">Resultados
              <select value={k} onChange={event => setK(Number(event.target.value))}>{[5, 10, 20, 50].map(value => <option key={value} value={value}>{value}</option>)}</select>
            </label>
          </div>
        </form>
      </section>

      {response && <PlanPanel plan={draftPlan ?? response.plan} editing={editingPlan} onEdit={() => setEditingPlan(true)} onChange={setDraftPlan} onCancel={() => { setDraftPlan(response.plan); setEditingPlan(false); }} onSave={savePlan} />}
      {error && <p className="notice error" role="alert">{error}</p>}
      {loading && <p className="notice">Preparando la búsqueda. La primera consulta CLAP puede demorar mientras se carga el modelo.</p>}

      {response && <>
        <div className="result-toolbar">
          <p><strong>{allResults.length}</strong> resultados · {response.took_ms} ms</p>
          {allResults.length > 1 && <button className="quiet" onClick={startContinuous}>{queue ? "Reproduciendo ranking…" : "▶ Reproducir todos"}</button>}
        </div>
        <Filters fileFilter={fileFilter} setFileFilter={setFileFilter} fromTime={fromTime} setFromTime={setFromTime} toTime={toTime} setToTime={setToTime} />
        <section className={`result-columns columns-${selectedIndexes.length}`} aria-label="Resultados">
          {selectedIndexes.map(index => <ResultColumn key={index} index={index} bucket={response.indexes[index]} fileFilter={fileFilter} fromTime={fromTime} toTime={toTime} focusedKey={focusedKey} setFocusedKey={setFocusedKey} requestPlay={requestPlay} registerPlayer={registerPlayer} onPlayerFinished={onPlayerFinished} />)}
        </section>
      </>}
    </main>
  );
}

function PlanPanel({ plan, editing, onEdit, onChange, onCancel, onSave }: { plan: SearchPlan; editing: boolean; onEdit: () => void; onChange: (value: SearchPlan) => void; onCancel: () => void; onSave: () => void }) {
  const copyPlan = async () => { await navigator.clipboard?.writeText(JSON.stringify(plan, null, 2)); };
  return <section className="plan" aria-label="Plan de búsqueda efectivo">
    <div><p className="eyebrow">Plan efectivo</p><p className="plan-hint">Este plan se puede copiar o reenviar para repetir el retrieval sin síntesis intermedia.</p></div>
    {editing ? <div className="plan-editor">
      {plan.indexes.includes("text") && <label>Query de texto<input value={plan.text_query ?? ""} onChange={event => onChange({ ...plan, text_query: event.target.value })} /></label>}
      {plan.indexes.includes("audio") && <><label>Descripción acústica<input value={plan.audio_query ?? ""} onChange={event => onChange({ ...plan, audio_query: event.target.value })} /></label><label>Query CLAP en inglés<input value={plan.audio_query_en ?? ""} onChange={event => onChange({ ...plan, audio_query_en: event.target.value })} /></label></>}
      <div className="inline-actions"><button className="primary" onClick={onSave}>Aplicar plan</button><button className="quiet" onClick={onCancel}>Cancelar</button></div>
    </div> : <div className="plan-summary">
      {plan.text_query && <span><b>Texto</b> “{plan.text_query}”</span>}
      {plan.audio_query && <span><b>Audio</b> “{plan.audio_query}”</span>}
      {plan.audio_query_en && <span className="translation">CLAP buscó en inglés: “{plan.audio_query_en}”</span>}
      {plan.rationale && <span className="rationale">{plan.rationale}</span>}
      <button className="quiet" onClick={() => void copyPlan()}>Copiar plan</button><button className="quiet" onClick={onEdit}>Editar plan</button>
    </div>}
  </section>;
}

function Filters({ fileFilter, setFileFilter, fromTime, setFromTime, toTime, setToTime }: { fileFilter: string; setFileFilter: (value: string) => void; fromTime: string; setFromTime: (value: string) => void; toTime: string; setToTime: (value: string) => void }) {
  return <div className="filters" aria-label="Filtros de resultados">
    <label>Archivo<input value={fileFilter} onChange={event => setFileFilter(event.target.value)} placeholder="Filtrar por nombre" /></label>
    <label>Desde (s)<input type="number" min="0" value={fromTime} onChange={event => setFromTime(event.target.value)} /></label>
    <label>Hasta (s)<input type="number" min="0" value={toTime} onChange={event => setToTime(event.target.value)} /></label>
  </div>;
}

function ResultColumn({ index, bucket, fileFilter, fromTime, toTime, focusedKey, setFocusedKey, requestPlay, registerPlayer, onPlayerFinished }: { index: SearchIndex; bucket?: IndexResults; fileFilter: string; fromTime: string; toTime: string; focusedKey: string | null; setFocusedKey: (key: string) => void; requestPlay: (audio: HTMLAudioElement) => void; registerPlayer: (key: string, command: PlayerCommand | null) => void; onPlayerFinished: (key: string) => void }) {
  const results = (bucket?.results ?? []).filter(result => {
    const matchesFile = result.original_file_name.toLocaleLowerCase().includes(fileFilter.toLocaleLowerCase());
    const afterStart = !fromTime || result.end_time >= Number(fromTime);
    const beforeEnd = !toTime || result.start_time <= Number(toTime);
    return matchesFile && afterStart && beforeEnd;
  });
  const topSimilarity = results[0]?.similarity ?? 0;
  const threshold = weakThreshold(index);
  return <section className="result-column">
    <header><p className="eyebrow">{labels[index]}</p><h2>{results.length} resultado{results.length === 1 ? "" : "s"}</h2>{bucket?.effective_query && <p className="effective-query">Query: “{bucket.effective_query}”</p>}{index === "audio" && bucket?.translated_query && <p className="effective-query">En inglés: “{bucket.translated_query}”</p>}</header>
    {!bucket ? <p className="empty">No se recibió una respuesta para este índice.</p> : !bucket.available ? <p className="notice">Índice no disponible en este dataset.{bucket.error ? ` ${bucket.error}` : ""}</p> : bucket.error ? <p className="notice error">{bucket.error}</p> : threshold !== undefined && results[0] && results[0].similarity < threshold ? <p className="notice">Coincidencias débiles: probablemente no haya buenos ejemplos en el corpus.</p> : null}
    {bucket?.available && !bucket.error && results.length === 0 && <p className="empty">No hay resultados que coincidan con estos filtros.</p>}
    <div className="cards">{results.map(result => <ResultCard key={resultKey(index, result)} result={result} index={index} topSimilarity={topSimilarity} focused={focusedKey === resultKey(index, result)} onFocus={() => setFocusedKey(resultKey(index, result))} requestPlay={requestPlay} registerPlayer={registerPlayer} onPlayerFinished={onPlayerFinished} />)}</div>
  </section>;
}

function ResultCard({ result, index, topSimilarity, focused, onFocus, requestPlay, registerPlayer, onPlayerFinished }: { result: SearchResult; index: SearchIndex; topSimilarity: number; focused: boolean; onFocus: () => void; requestPlay: (audio: HTMLAudioElement) => void; registerPlayer: (key: string, command: PlayerCommand | null) => void; onPlayerFinished: (key: string) => void }) {
  const key = resultKey(index, result);
  const citation = `${result.original_file_name} · ${asTime(result.start_time)}–${asTime(result.end_time)} · segment_id ${result.segment_id} · índice ${result.search_index_label ?? labels[index]}`;
  const normalized = topSimilarity > 0 ? Math.max(0, Math.min(100, (result.similarity / topSimilarity) * 100)) : 0;
  const copyCitation = async () => { await navigator.clipboard?.writeText(citation); };
  return <article className={`result-card${focused ? " is-focused" : ""}`} tabIndex={0} onFocus={onFocus}>
    <div className="card-top"><span className="rank">#{result.rank}</span><div className="score" title="Similitud coseno. Los valores de CLAP y de texto no son comparables entre sí."><span><i style={{ width: `${normalized}%` }} /></span><small>{result.similarity.toFixed(3)}</small></div></div>
    <h3 title={result.original_file_name}>{result.original_file_name}</h3>
    <p className="timestamp">{asTime(result.start_time)} → {asTime(result.end_time)} <span>({Math.round(result.duration ?? result.end_time - result.start_time)} s)</span></p>
    {result.clip_url ? <AudioPlayer id={key} result={result} preload={result.rank === 1} requestPlay={requestPlay} registerPlayer={registerPlayer} onFinished={() => onPlayerFinished(key)} /> : <p className="missing-audio">Clip no disponible para este segmento.</p>}
    <p className="transcript">“{result.text || "Sin transcripción disponible."}”</p>
    {index === "audio" && result.yamnet_top_classes?.length ? <p className="yamnet"><b>AudioSet</b> {result.yamnet_top_classes.slice(0, 3).map(item => `${item.label ?? item.class_name ?? "event"} ${item.score.toFixed(2)}`).join(" · ")}</p> : null}
    <div className="card-actions"><button className="quiet" onClick={() => void copyCitation()}>Copiar cita</button>{result.clip_url && <a className="quiet" href={result.clip_url} download={`segment_${result.segment_id}.opus`}>Descargar</a>}<span title={citation}>ID {result.segment_id}</span></div>
  </article>;
}

function AudioPlayer({ id, result, preload, requestPlay, registerPlayer, onFinished }: { id: string; result: SearchResult; preload: boolean; requestPlay: (audio: HTMLAudioElement) => void; registerPlayer: (key: string, command: PlayerCommand | null) => void; onFinished: () => void }) {
  const audio = useRef<HTMLAudioElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const [context, setContext] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const clipStart = result.clip_start_time ?? result.start_time;
  const clipEnd = result.clip_end_time ?? result.end_time;
  const rangeStart = context ? Math.max(clipStart, result.start_time - 10) : result.start_time;
  const rangeEnd = context ? Math.min(clipEnd, result.end_time + 10) : result.end_time;
  const startOffset = Math.max(0, rangeStart - clipStart);
  const endOffset = Math.max(startOffset + 0.1, rangeEnd - clipStart);
  const progress = Math.max(0, Math.min(100, ((position - startOffset) / (endOffset - startOffset)) * 100));
  const play = useCallback(() => {
    const element = audio.current;
    if (!element) return;
    requestPlay(element);
    if (element.currentTime < startOffset || element.currentTime >= endOffset) element.currentTime = startOffset;
    void element.play().catch(() => setPlaying(false));
  }, [endOffset, requestPlay, startOffset]);
  useEffect(() => { registerPlayer(id, { play, focus: () => root.current?.focus() }); return () => registerPlayer(id, null); }, [id, play, registerPlayer]);
  useEffect(() => { const element = audio.current; if (element) { element.pause(); element.currentTime = startOffset; } setPlaying(false); setPosition(startOffset); }, [startOffset, endOffset]);
  const toggle = () => { if (audio.current?.paused) play(); else audio.current?.pause(); };
  return <div className="audio-player" ref={root} tabIndex={-1}>
    <audio ref={audio} preload={preload ? "auto" : "metadata"} src={`${result.clip_url}#t=${startOffset},${endOffset}`} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onTimeUpdate={event => { const time = event.currentTarget.currentTime; setPosition(time); if (time >= endOffset) { event.currentTarget.pause(); onFinished(); } }} onEnded={onFinished} />
    <button className="play" onClick={toggle} aria-label={playing ? "Pausar segmento" : "Reproducir segmento"}>{playing ? "❚❚" : "▶"}</button>
    <input aria-label="Progreso de audio" type="range" min={startOffset} max={endOffset} step="0.1" value={Math.min(Math.max(position, startOffset), endOffset)} style={{ "--progress": `${progress}%` } as CSSProperties} onChange={event => { const value = Number(event.target.value); if (audio.current) audio.current.currentTime = value; setPosition(value); }} />
    <time>{asTime(Math.max(0, position - startOffset))}</time>
    <button className={`context ${context ? "active" : ""}`} onClick={() => setContext(value => !value)} title="Extiende la escucha diez segundos a cada lado">±10 s</button>
  </div>;
}
