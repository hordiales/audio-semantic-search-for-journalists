export type SearchIndex = "text" | "audio";
export type IndexSelection = SearchIndex | "both";

export interface SearchPlan {
  original_query: string;
  indexes: SearchIndex[];
  text_query?: string;
  audio_query?: string;
  audio_query_en?: string;
  rationale?: string;
}

export interface AudioClass {
  label?: string;
  class_name?: string;
  score: number;
}

export interface SearchResult {
  segment_id: number;
  text: string;
  start_time: number;
  end_time: number;
  duration?: number;
  original_file_name: string;
  language?: string;
  confidence?: number;
  similarity: number;
  rank: number;
  search_index: SearchIndex;
  search_index_label?: string;
  clip_url?: string;
  clip_start_time?: number;
  clip_end_time?: number;
  clip_expires_at?: string;
  yamnet_top_classes?: AudioClass[];
}

export interface IndexResults {
  available: boolean;
  effective_query?: string;
  translated_query?: string;
  results: SearchResult[];
  error?: string;
}

export interface SearchResponse {
  query: string;
  plan: SearchPlan;
  took_ms: number;
  indexes: Partial<Record<SearchIndex, IndexResults>>;
}

export interface SearchRequest {
  query: string;
  indexes: SearchIndex[];
  k: number;
  rewrite: boolean;
  plan?: SearchPlan;
}
