// The wire format of research_copilot.api: `streaming.astream_run`'s events,
// plus the API's own `thread` event, which always comes first.

export type AgentName = 'supervisor' | 'researcher' | 'writer' | 'critic';
export const AGENTS: AgentName[] = ['supervisor', 'researcher', 'writer', 'critic'];

export interface Policy {
  routing?: 'supervisor' | 'fixed';
  enable_critic?: boolean;
  require_approval?: boolean;
  max_revisions?: number;
  max_research_iterations?: number;
  enable_planning?: boolean;
}

export type CitationStatus = 'found' | 'not_found' | 'invalid' | 'error';
export interface CitationCheck {
  arxiv_id: string;
  status: CitationStatus;
}

export interface ReviewPayload {
  question: string;
  mode: string;
  draft: string;
  critique: string;
  prompt?: string;
}

export type RunEvent =
  | { type: 'thread'; thread_id: string; is_new: boolean; policy: Policy }
  | { type: 'node'; phase: 'start' | 'end'; agent: string; node: string; depth: number }
  | {
      type: 'decision';
      step: number;
      proposed: string | null;
      routed_to: string;
      override: string;
      rationale: string;
      brief: string;
    }
  | { type: 'verdict'; verdict: string; critique: string; citation_checks: CitationCheck[] }
  | {
      type: 'intervention';
      node?: string;
      kind?: string;
      part?: string;
      detail?: string;
      tokens_before?: number;
      tokens_after?: number;
      limit?: number;
    }
  | { type: 'token'; agent: string; text: string }
  | { type: 'tool'; phase: 'start' | 'end'; agent: string; name: string; arg?: string }
  | { type: 'paused'; payload: ReviewPayload }
  | {
      type: 'done';
      answer: string;
      dispatches: Record<string, number>;
      revisions: number;
      verdict: string;
      unsupported_citations: string[];
      interventions: number;
    }
  | { type: 'stopped'; reason: string; detail: string }
  | { type: 'error'; kind: string; detail: string };

export interface RunRequest {
  question: string;
  mode: 'live-search' | 'knowledge-base';
  thread_id?: string | null;
  routing: 'supervisor' | 'fixed';
  critic: boolean;
  approve: boolean;
  plan: boolean;
  max_revisions: number;
}

export interface ReviewRequest {
  decision: 'approve' | 'reject' | 'edit';
  text?: string;
  note?: string;
}

export interface Health {
  status: string;
  provider: string;
  model: string;
  api_key_set: boolean;
  checkpointer: string;
  defaults: { max_revisions: number; critic: boolean; plan: boolean };
}

export interface ThreadSummary {
  thread_id: string;
  question: string;
  turns: number;
  updated_at: string | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ThreadView {
  thread_id: string;
  question: string;
  mode: string;
  policy: Policy;
  messages: ChatMessage[];
  pending_review: ReviewPayload | null;
}
